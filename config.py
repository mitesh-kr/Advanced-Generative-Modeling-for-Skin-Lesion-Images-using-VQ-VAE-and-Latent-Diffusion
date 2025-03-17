import torch
import random
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
seed = 42
acc_steps = 1
image_save_steps = 5
num_epochs_vqvae = 10
num_epochs_diffusion = 100
lr = 0.0001
disc_step_start = 1000
batch_size = 64

# Model parameters
vqvae_codebook_size = 8192
vqvae_embedding_dim = 3
norm_channels = 32
num_heads = 4

# Diffusion parameters
num_timesteps = 1000
beta_start = 0.0015
beta_end = 0.0195

# Set random seeds for reproducibility
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
