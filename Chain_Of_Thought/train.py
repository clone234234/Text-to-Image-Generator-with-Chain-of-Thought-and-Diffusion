import torch
import torch.nn as nn 
from torch.nn import functional as F
from transformer import Transformer


def preprocess_data():
    with open("D:\genpic\Chain_Of_Thought\input.txt",'r',encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    processed_lines = []
    for line in lines:
        filtered_line = ''.join(char for char in line if char.isalnum() or char in ' .,?!-:;')
        if filtered_line:  
            processed_lines.append(filtered_line)
    special_tokens = ['<pad>', '<unk>']
    chars= sorted(list(set(''.join(processed_lines))))
    all_tokens = special_tokens + chars
    vocab= {char: idx for idx, char in enumerate(all_tokens)}        
    return processed_lines, vocab, len(vocab)

def generate_mask(sz, device):
    mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask



def batch(batch_data, vocab, device, max_seq_length=32):  

    src_seqs = [line for line in batch_data]
    tgt_seqs = [line for line in batch_data]
    pad_idx = vocab['<pad>']

    processed_src = []
    processed_tgt = []
    
    for src_seq, tgt_seq in zip(src_seqs, tgt_seqs):
        src_indices = [vocab.get(char, vocab['<unk>']) for char in src_seq]  
        tgt_indices = [vocab.get(char, vocab['<unk>']) for char in tgt_seq]
        
        if len(src_indices) > max_seq_length:
            src_indices = src_indices[:max_seq_length]
        
        if len(tgt_indices) > max_seq_length:
            tgt_indices = tgt_indices[:max_seq_length]

        src_padding = [pad_idx] * (max_seq_length - len(src_indices))
        tgt_padding = [pad_idx] * (max_seq_length - len(tgt_indices))
        
        processed_src.append(src_indices + src_padding)
        processed_tgt.append(tgt_indices + tgt_padding)
    
    src_tensor = torch.tensor(processed_src, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(processed_tgt, dtype=torch.long, device=device)
    
    return src_tensor, tgt_tensor



def train_transformer(model, data, vocab, num_epochs=30, batch_size=16, device='cuda'):  
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    pad_idx = vocab['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='mean', label_smoothing=0.1) 
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches_processed = 0
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            src_tensor, tgt_tensor = batch(batch_data, vocab, device, max_seq_length=32) 
            src_keep_mask = (src_tensor != pad_idx).unsqueeze(1).unsqueeze(2) 

            seq_len_tgt_full = tgt_tensor.size(1)

            tgt_padding_keep_mask_full = (tgt_tensor != pad_idx).unsqueeze(1).unsqueeze(2) 

            look_ahead_mask_float = generate_mask(seq_len_tgt_full, device)

            look_ahead_keep_mask_bool = (look_ahead_mask_float != float('-inf')).unsqueeze(0).unsqueeze(0) 

            tgt_keep_mask_full = tgt_padding_keep_mask_full & look_ahead_keep_mask_bool 

            tgt_mask_for_decoder_self_attn = tgt_keep_mask_full[:, :, :-1, :-1] 

            optimizer.zero_grad()

            if torch.isnan(src_tensor).any() or torch.isnan(tgt_tensor).any():
                print(f"NaN detected in input tensors at batch {i//batch_size}")
                continue
                
            output = model(src_tensor, tgt_tensor[:, :-1], src_keep_mask, tgt_mask_for_decoder_self_attn)

            if torch.isnan(output).any():
                print(f"NaN detected in model output at batch {i//batch_size}")
                continue

            output_flat = output.reshape(-1, output.size(-1))
            target_for_loss_flat = tgt_tensor[:, 1:].reshape(-1)

            non_pad_mask_for_loss = (target_for_loss_flat != pad_idx)
            num_valid_targets = non_pad_mask_for_loss.sum().item()

            if num_valid_targets > 0:
                loss = criterion(output_flat, target_for_loss_flat)

                if torch.isnan(loss).any():
                    print(f"NaN detected in loss at batch {i//batch_size}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient detected in {name}")
                        has_nan_grad = True
                        break
                
                if not has_nan_grad:
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches_processed += 1
                else:
                    optimizer.zero_grad()
            else:
                pass

        if num_batches_processed > 0:
            avg_loss = total_loss / num_batches_processed
            scheduler.step(avg_loss)  
            print(f'Epoch {epoch+1}/{num_epochs}, Processed Batches: {num_batches_processed}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: No batches were processed (all targets might have been padding).')


if __name__ == "__main__":
    lines, vocab, vocab_size = preprocess_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,      
        num_heads=8,
        d_ff=1024,          
        num_layers=4,       
        dropout=0.1,
        max_seq_length=256 
    ).to(device)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.1)
    
    model.apply(init_weights)
    
    print("Starting training...")
    train_transformer(model, lines, vocab, num_epochs=30, batch_size=16, device=device)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vocab_size': vocab_size,
        'model_config': {
            'src_vocab_size': vocab_size,
            'tgt_vocab_size': vocab_size,
            'd_model': 256,
            'num_heads': 8,
            'd_ff': 1024,
            'num_layers': 4,
            'dropout': 0.1,
            'max_seq_length': 256
        }
    }
    
    torch.save(checkpoint, 'D:\\project-2\\Chain_Of_Thought\\model1.pth')

    with open('vocab.txt', 'w', encoding='utf-8') as f:
        f.write(str(vocab))
    
    print(f"Model and vocabulary saved successfully!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: <pad>={vocab['<pad>']}, <unk>={vocab['<unk>']}")