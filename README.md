# VQ-VAE Diffusion Model

This repository implements a two-stage generative model combining Vector Quantized Variational Autoencoder (VQ-VAE) with a diffusion model. The approach first compresses images into a discrete latent space using VQ-VAE, then learns a diffusion probabilistic model in this latent space for high-quality image generation.


## Google Colab Link:(https://colab.research.google.com/drive/1AZEWxF4qpbvdl8YslnA-48lTzKRdJ-ja?usp=sharing)


## Architecture

The project consists of two main components:

1. **VQ-VAE**: A neural compression model that maps images to a discrete latent space.
   - Encoder: Transforms images into a continuous latent representation
   - Vector Quantization: Maps continuous latents to a codebook of discrete embeddings
   - Decoder: Reconstructs images from the quantized latent representations

2. **Diffusion Model**: A denoising diffusion probabilistic model that operates in the VQ-VAE's latent space.
   - Gradually adds noise to latent codes according to a fixed schedule
   - Learns to reverse this process to generate new latent codes
   - Generated latent codes can be decoded through the VQ-VAE decoder

## Project Structure

```
├── README.md
├── requirements.txt
├── config.py
├── data/
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── vqvae.py
│   ├── discriminator.py
│   └── diffusion.py
├── train/
│   ├── __init__.py
│   ├── train_vqvae.py
│   └── train_diffusion.py
├── utils/
│   ├── __init__.py
│   └── schedulers.py
└── infer.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mitesh-kr/Advanced-Generative-Modeling-for-Skin-Lesion-Images-using-VQ-VAE-and-Latent-Diffusion.git
cd Advanced-Generative-Modeling-for-Skin-Lesion-Images-using-VQ-VAE-and-Latent-Diffusion

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

The training process has two stages:

### 1. Training the VQ-VAE

```bash
python -m train.train_vqvae --data_path /path/to/dataset --batch_size 64 --num_epochs 100
```

### 2. Training the Diffusion Model

```bash
python -m train.train_diffusion --vqvae_checkpoint path/to/vqvae_checkpoint.pth --data_path /path/to/dataset --batch_size 64 --num_epochs 100
```

## Inference

To generate new images:

```bash
python infer.py --vqvae_checkpoint path/to/vqvae_checkpoint.pth --diffusion_checkpoint path/to/diffusion_checkpoint.pth --num_samples 16 --output_dir samples
```

## Parameters

The model behavior can be configured in `config.py`. Key parameters include:

- **VQ-VAE**:
  - `embedding_dim`: Dimension of the codebook embeddings
  - `num_embeddings`: Number of vectors in the codebook
  - `commitment_cost`: Weight of the commitment loss

- **Diffusion Model**:
  - `num_timesteps`: Number of diffusion steps
  - `beta_start`: Starting value for variance schedule
  - `beta_end`: Ending value for variance schedule

## Results

Sample generated images:

[Genearted sample images here](https://drive.google.com/file/d/19GDyU2i3uuA1o7BjWorm9nsIHjV6-t8Y/view?usp=sharing)

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request
