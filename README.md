# Text-to-Image Generator with Chain-of-Thought and Diffusion

## Overview

This project implements an AI pipeline that integrates:

- **Chain-of-Thought (CoT) Model**: Enhances user prompts by generating detailed and contextually rich text descriptions.
- **Diffusion Model (DDPM)**: Generates images from refined text prompts.
- **Context Window Extension**: Supports multi-turn conversations by retaining dialogue history.

---

## Directory Structure

```
.
├── pipeline.py
├── Chain_Of_Thought/
│   └── COT_text_gen.py
├── diffusion_model/
│   ├── pipeline.py
│   ├── clip.py
│   ├── diffusion.py
│   ├── ddpm.py
│   ├── VAen_decoder.py
│   └── data/
│       └── v1-5-pruned-emaonly.ckpt   # Stable Diffusion weights
├── model.pth
├── diffusion_model.pth
├── vae_encoder.pth
├── vae_decoder.pth
├── clip_model.pth
├── output.png
```

---

## Getting Started

### 1. Environment Setup

```bash
pip install torch torchvision transformers tqdm
```
### 2.  Download Pretrained Weights
Stable Diffusion v1.5 Checkpoint:
Download from Hugging Face(https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)

Save file v1-5-pruned-emaonly.ckpt into:
```bash
diffusion_model/data/v1-5-pruned-emaonly.ckpt
```
### 3. Model Initialization

```python
from COT_text_gen import ImprovedChainOfThought
from pipeline import generate
from combined_pipeline import TextToImagePipeline

# Initialize the Chain-of-Thought model
text_model = ImprovedChainOfThought(...)

# Initialize the text-to-image pipeline
pipe = TextToImagePipeline(
    text_gen_model=text_model,
    diffusion_pipeline=generate,
    device='cuda'
)
```

### 3. Generating Images

```python
image = pipe("Draw a cat studying", seed=42)
image.save("output.png")
```

---

## Key Features

- **Prompt Enhancement**: Automatically refines vague user prompts using Chain-of-Thought reasoning.
- **High-Quality Image Generation**: Utilizes a diffusion-based UNet model with DDPM sampling.
- **Conversation Memory**: Supports multi-turn dialogues by maintaining context history.
- **Configurable Parameters**: Adjustable settings for `n_steps`, `strength`, and more.

---

## Model Explanation

This project follows a two-stage AI pipeline:

1. **Text Understanding & Enhancement (Chain-of-Thought Model)**
   - The input text from the user is processed by the `ImprovedChainOfThought` transformer.
   - The model expands the prompt with richer context and more descriptive details.  
     *Example*: "Draw a cat" → "A fluffy white cat sitting on a wooden table, sunlight streaming through a window."
   - This step ensures the diffusion model receives more detailed guidance for image generation.

2. **Image Generation (Diffusion + UNet + VAE)**
   - **Text Encoding**: The enhanced prompt is transformed into an embedding vector (using the internal text encoder of the pipeline).
   - **Noise Initialization**: The diffusion process starts from pure Gaussian noise.
   - **UNet Denoising**: The UNet predicts the noise at each timestep, guided by the text embedding.
   - **DDPM Sampling**: The sampler gradually denoises the latent representation into a meaningful image.
   - **VAE Decoding**: The latent image is decoded into full-resolution pixel space.

**Data Flow Diagram:**
```
User Prompt
   ↓
Chain-of-Thought Transformer
   ↓
Enhanced Prompt → Text Encoder → Diffusion UNet + DDPM Sampler
   ↓
VAE Decoder → Final Image
```

**Why This Architecture Works:**
- The Chain-of-Thought stage ensures vague prompts are turned into vivid descriptions, improving the guidance for the diffusion model.
- UNet with self-attention and cross-attention layers captures both global composition and fine details.
- The VAE enables computation in a smaller latent space, making training and inference faster without significant quality loss.

---

## Notes

- **Hardware Requirements**: A GPU with at least 8GB VRAM is recommended for efficient diffusion model inference.
- **Data Files**: Ensure `vocab.txt` (CoT vocabulary) and `input.txt` (test inputs) are available in the project directory.

---

## Author

- **Developer**: Bùi Nguyễn Gia BẢO 

