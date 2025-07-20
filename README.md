# Text-to-Image Generator with Chain-of-Thought and Diffusion

## Overview

This project implements an AI pipeline that integrates:

- **Chain-of-Thought (CoT) Model**: Enhances user prompts by generating detailed and contextually rich text descriptions.
- **Diffusion Model (DDPM)**: Generates high-quality images from refined text prompts.
- **Context Window Extension**: Supports multi-turn conversations by retaining dialogue history.

---

## Directory Structure

```
project/
├── models/
│   ├── attention.py         # Self-Attention and Cross-Attention mechanisms
│   ├── transformer.py       # Transformer model for Chain-of-Thought
│   ├── diffusion.py         # UNet and noise prediction logic
│   └── VAen_decoder.py      # Variational Autoencoder (VAE) encoder and decoder
├── sampler/
│   └── ddpm.py              # Denoising Diffusion Probabilistic Model (DDPM) sampling logic
├── COT_text_gen.py          # ImprovedChainOfThought model for text generation
├── pipeline.py              # Image generation function (generate)
├── train.py                 # Training script for the diffusion model
├── vocab.txt                # Vocabulary file for the CoT model
├── input.txt                # Sample test input file
├── combined_pipeline.py     # Integrated pipeline combining CoT and diffusion
└── README.md                # Project documentation
```

---

## Getting Started

### 1. Environment Setup

Install the required dependencies:

```bash
pip install torch torchvision transformers tqdm
```

### 2. Model Initialization

```python
from COT_text_gen import ImprovedChainOfThought
from pipeline import generate
from combined_pipeline import TextToImagePipeline
from transformers import CLIPTokenizer

# Load the CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Initialize the Chain-of-Thought model
text_model = ImprovedChainOfThought(...)

# Initialize the text-to-image pipeline
pipe = TextToImagePipeline(
    text_gen_model=text_model,
    diffusion_pipeline=generate,
    tokenizer=tokenizer,
    device='cuda'
)
```

### 3. Generating Images

Generate an image from a simple text prompt:

```python
image = pipe("Draw a cat studying", seed=42)
image.save("output.png")
```

---

## Key Features

- **Prompt Enhancement**: Automatically refines vague user prompts using Chain-of-Thought reasoning.
- **High-Quality Image Generation**: Utilizes a diffusion-based UNet model with DDPM sampling.
- **Conversation Memory**: Supports multi-turn dialogues by maintaining context history.
- **Configurable Parameters**: Adjustable settings for `cfg_scale`, `n_steps`, `strength`, and more.

---

## Model Components

| Component                  | Description                                      |
|---------------------------|--------------------------------------------------|
| `ImprovedChainOfThought`  | Generates text with reasoning-based augmentation |
| `Diffusion` + `UNet`      | Predicts and removes noise for image generation  |
| `DDPMSampler`             | Implements denoising for diffusion process       |
| `VAE`                     | Encodes images into a latent space              |
| `Self/Cross Attention`    | Enhances UNet and text encoding capabilities    |

---

## Advanced Usage

### Multi-Turn Conversation Example

```python
pipe.context_memory = [
    "User: Draw a soldier in a forest.",
    "Bot: Alright, I'll create an image of a soldier hiding in a forest.",
    "User: Now add a sunrise.",
]

img = pipe("Create a version with morning mist.")
img.show()
```

---

## Notes

- **Hardware Requirements**: A GPU with at least 8GB VRAM is recommended for efficient diffusion model inference.
- **Training**: To retrain the diffusion model, use the `train.py` script.
- **Data Files**: Ensure `vocab.txt` (CoT vocabulary) and `input.txt` (test inputs) are available in the project directory.

---

## Author

- **Developer**: Bui Nguyen Gia Bao
