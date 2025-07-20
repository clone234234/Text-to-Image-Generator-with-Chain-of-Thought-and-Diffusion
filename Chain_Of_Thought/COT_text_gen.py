import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
from transformer import Transformer
from transformers import CLIPTokenizer

class ImprovedChainOfThought(nn.Module):
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        idx_to_char: Optional[Dict[int, str]] = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        tokenizer: Optional[CLIPTokenizer] = None
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vocab_size = tokenizer.vocab_size if tokenizer else len(vocab)
        self.transformer = Transformer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            vocab=vocab,
            vocab_size=vocab_size
        ).to(self.device)
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.idx_to_char = idx_to_char
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pad_token = tokenizer.pad_token_id if tokenizer else vocab.get('<pad>', 0)
        self.unk_token = tokenizer.unk_token_id if tokenizer else vocab.get('<unk>', 1)
        self.eos_token = tokenizer.eos_token_id if tokenizer else vocab.get('<eos>')
        if self.eos_token is None:
            raise ValueError("'<eos>' token not found in vocabulary or tokenizer")
        self.cot_templates = {
            'step_by_step': " Let's think step by step:",
            'analysis': " Let me analyze this:",
            'reasoning': " Here's my reasoning:",
            'problem_solving': " Let me break down this problem:",
        }
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)
    
    def create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens != self.pad_token).unsqueeze(1).unsqueeze(2)
    
    def encode_text(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(self.device)
            indices = tokens[0].tolist()
        else:
            if self.vocab is None:
                raise ValueError("Custom vocabulary not provided")
            indices = [self.vocab.get(char, self.unk_token) for char in text]
            if len(indices) > self.max_seq_length - 10:
                indices = indices[:self.max_seq_length - 10]
            if not indices:
                raise ValueError("No valid tokens found in input")
            tokens = torch.tensor([indices], dtype=torch.long, device=self.device)
        return tokens, indices
    
    def decode_indices(self, indices: List[int]) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(indices, skip_special_tokens=True)
        return ''.join([self.idx_to_char.get(idx, '<unk>') for idx in indices])
    
    def apply_temperature_sampling(self, logits: torch.Tensor, temperature: float = 1.0, 
                                 top_k: int = 50, top_p: float = 0.9) -> int:
        if temperature <= 0:
            return torch.argmax(logits).item()
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()
    
    def remember_chat_history(self, question: str, chat_history: Optional[List[str]] = None, 
                            cot_type: str = 'step_by_step') -> str:
        if chat_history is None:
            chat_history = []
        cot_prompt = self.cot_templates.get(cot_type, self.cot_templates['step_by_step'])
        current_prompt = question + cot_prompt
        current_prompt_length = len(current_prompt)
        available_history_length = self.max_seq_length - current_prompt_length - 50
        if available_history_length <= 0:
            return current_prompt
        history_context = ''
        total_history_length = 0
        for i in range(len(chat_history) - 1, -1, -1):
            entry = chat_history[i]
            entry_length = len(entry)
            if total_history_length + entry_length > available_history_length:
                break
            history_context = entry + "\n" + history_context
            total_history_length += entry_length + 1
        if history_context.strip():
            full_context = history_context.strip() + "\n" + current_prompt
        else:
            full_context = current_prompt
        return full_context
    
    def generate_with_cot(self, question: str, cot_type: str = 'step_by_step', 
                         max_length: int = 200, temperature: float = 0.8,
                         top_k: int = 50, top_p: float = 0.9) -> str:
        try:
            self.eval()
            cot_prompt = question + self.cot_templates.get(cot_type, self.cot_templates['step_by_step'])
            input_tensor, input_indices = self.encode_text(cot_prompt)
            generated_indices = input_indices.copy()
            with torch.no_grad():
                for step in range(max_length):
                    if input_tensor.size(1) >= self.max_seq_length - 1:
                        break
                    seq_len = input_tensor.size(1)
                    src_mask = self.create_padding_mask(input_tensor)
                    tgt_mask = self.create_causal_mask(seq_len)
                    try:
                        output = self.transformer(input_tensor, input_tensor, src_mask, tgt_mask)
                        next_token_logits = output[0, -1, :]
                        if not torch.isfinite(next_token_logits).all():
                            print(f"Warning: Invalid logits at step {step}")
                            break
                        next_token_idx = self.apply_temperature_sampling(
                            next_token_logits, temperature, top_k, top_p
                        )
                        generated_indices.append(next_token_idx)
                        next_token_tensor = torch.tensor([[next_token_idx]], 
                                                       dtype=torch.long, device=self.device)
                        input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                        if next_token_idx == self.eos_token:
                            break
                        if len(generated_indices) > 20:
                            recent_tokens = generated_indices[-10:]
                            if len(set(recent_tokens)) <= 2:
                                break
                    except Exception as e:
                        print(f"Error during generation at step {step}: {e}")
                        break
            generated_text = self.decode_indices(generated_indices)
            return generated_text
        except Exception as e:
            print(f"Error in generate_with_cot: {e}")
            return question
    
    def forward(self, question: str, max_length: int = 200, temperature: float = 0.8) -> str:
        return self.generate_with_cot(question, max_length=max_length, temperature=temperature)

def load_improved_model(model_path: str = 'model.pth', device: str = 'auto') -> Tuple[Optional['ImprovedChainOfThought'], Optional[Dict]]:
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            vocab = checkpoint.get('vocab')
            vocab_size = checkpoint.get('vocab_size')
            model_config = checkpoint.get('model_config', {})
            idx_to_char = {idx: char for char, idx in vocab.items()} if vocab else None
            tokenizer = None
            if checkpoint.get('use_tokenizer', False):
                tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
            model = ImprovedChainOfThought(
                vocab=vocab,
                idx_to_char=idx_to_char,
                d_model=model_config.get('d_model', 256),
                num_heads=model_config.get('num_heads', 8),
                num_layers=model_config.get('num_layers', 4),
                d_ff=model_config.get('d_ff', 1024),
                dropout=model_config.get('dropout', 0.1),
                max_seq_length=model_config.get('max_seq_length', 256),
                tokenizer=tokenizer
            )
            model.transformer.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print(f"✓ Successfully loaded improved model")
            print(f"  Vocabulary size: {vocab_size}")
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Device: {device}")
            return model, vocab
        else:
            print("✗ Checkpoint format not recognized")
            return None, None
    except FileNotFoundError:
        print(f"✗ Model file '{model_path}' not found")
        return None, None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def test_improved_generation():
    model, vocab = load_improved_model()
    if model is None:
        print("Failed to load model for testing")
        return
    test_cases = [
        ("What is 2 + 2?", "step_by_step"),
        ("Why is the sky blue?", "analysis"),
        ("How do you make a sandwich?", "problem_solving"),
        ("What causes rain?", "reasoning")
    ]
    print("\n" + "="*60)
    print("TESTING IMPROVED CHAIN-OF-THOUGHT GENERATION")
    print("="*60)
    for question, cot_type in test_cases:
        print(f"\nQuestion: {question}")
        print(f"CoT Type: {cot_type}")
        print("-" * 40)
        try:
            result = model.generate_with_cot(
                question=question,
                cot_type=cot_type,
                max_length=150,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            print(f"Generated: {result}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)

if __name__ == "__main__":
    test_improved_generation()