
# 🧠💡 Text-to-Image Generation Pipeline using Chain-of-Thought and Diffusion Models

This project implements a **Text-to-Image Generation Pipeline** that first enhances user prompts through a **Chain-of-Thought (CoT) reasoning model**, and then generates corresponding images using a **Diffusion-based generative model**. The pipeline is designed for creative applications such as storytelling, concept visualization, and generative AI tasks.

---

## 📌 Features

- **ImprovedChainOfThought** model for expanding prompts using step-by-step reasoning.
- **CLIP-based tokenizer and encoder** to transform text into latent space.
- **VAE Encoder–Decoder** structure for image representation and generation.
- **DDPM Sampler** (Denoising Diffusion Probabilistic Models) for high-quality image synthesis.
- Configurable pipeline: supports CFG scale, strength, inference steps, and random seed control.

---

## 🔍 Model Explanation

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **CoT Model**    | A Transformer-based model that generates detailed prompts using Chain-of-Thought reasoning to improve image relevance. |
| **CLIP**         | Converts text into a latent embedding space compatible with the diffusion model. |
| **VAE Encoder**  | Compresses image features into a latent representation.                     |
| **VAE Decoder**  | Reconstructs images from latent vectors after sampling.                     |
| **DDPM**         | Diffusion-based image generator that denoises latent space into realistic images. |

---

## 🧩 Architecture Overview

```
User Prompt
   ↓
[Chain-of-Thought Reasoning]
   ↓
Refined Prompt (e.g., “A dog eating a hotdog in Central Park”)
   ↓
[CLIP Tokenizer & Encoder]
   ↓
[Diffusion-based Image Generator]
   ↓
Generated Image Output (RGB)
```

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install torch torchvision transformers Pillow
```

### 2. Download Pretrained Models

#### 📥 Get Stable Diffusion v1.5 Checkpoint

- Visit: [Stable Diffusion v1-5 Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)
- Download file: `v1-5-pruned-emaonly.ckpt`
- Save it to:  
  ```bash
  diffusion_model/data/v1-5-pruned-emaonly.ckpt
  ```

### 3. Required Model Files

Ensure the following files exist:

```
model.pth                   # Chain-of-Thought model checkpoint
diffusion_model.pth         # DDPM model weights
vae_encoder.pth             # VAE encoder weights
vae_decoder.pth             # VAE decoder weights
clip_model.pth              # CLIP model weights
```

> ⚠️ If these files are missing, the pipeline will fallback to untrained models and results may be poor.

---

## 🚀 Running the Pipeline

```bash
python pipeline.py
```

This will:
- Accept a user prompt
- Expand it using Chain-of-Thought reasoning
- Generate an image using the diffusion model
- Save the image as `output.png`

---

## 📁 File Structure

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
├── output.png
```

---

## 📚 License

This project is intended for **research and educational purposes** only. Contact the author for commercial use.

---

## 👨‍💻 Author

**[Your Name]**  
AI Researcher / Developer  
📫 [your-email@example.com] | GitHub: [your-profile]
