import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from VAen_decoder import encoder, decoder
from tqdm import tqdm
from ddpm import DDPMSampler
width = 512
height = 512
latent_height = height // 8
latent_width = width // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("Strength must be in the range (0, 1]")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            condition_token = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            condition_token = torch.tensor(condition_token, dtype=torch.long, device=device)
            condition_context = clip(condition_token)

            uncond_token = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_token = torch.tensor(uncond_token, dtype=torch.long, device=device)
            uncondi_context = clip(uncond_token)

            context = torch.cat([uncondi_context, condition_context])
        else:
            token = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            token = torch.tensor(token, dtype=torch.long, device=device)
            context = clip(token)
        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        latent_shape = (1, 4, latent_height, latent_width)

        if input_image is not None:
            encode = models['encoder']
            encode.to(device)

            input_image_tensor = input_image.resize((width, height))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            latents = encode(input_image_tensor, encoder_noise)

            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encode)
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion_model = models['diffusion']
        diffusion_model.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion_model(model_input, context, time_embedding)

            if do_cfg:
                output_uncond, output_cond = model_output.chunk(2)
                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion_model)

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]





def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)