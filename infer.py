import os
import torch
import argparse
import torchvision
from torchvision.utils import make_grid
from models.vqvae import VQVAE
from models.diffusion import UNet
from utils.schedulers import LinearNoiseScheduler
from config import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples from trained VQVAE-Diffusion model')
    parser.add_argument('--vqvae_checkpoint', type=str, required=True, help='Path to the VQVAE checkpoint')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True, help='Path to the Diffusion model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='samples', help='Directory to save generated samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_samples(vqvae, diffusion_model, scheduler, num_samples, batch_size, device, output_dir):
    """Generate samples using the trained VQVAE and diffusion model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples in batches
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            actual_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            print(f"Generating batch {batch_idx+1}/{num_batches} with {actual_batch_size} samples...")
            
            # Start with random noise in latent space
            latent_size = 4  # This is typically downsampled by factor of 2^3 from image size
            xt = torch.randn((actual_batch_size, 3, latent_size, latent_size)).to(device)
            
            # Reverse diffusion process (simplified version for demonstration)
            # In a full implementation, you'd iterate backwards through all timesteps
            noise_pred = diffusion_model(xt, torch.tensor([0]).to(device))
            
            # Get the predicted clean latent
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.tensor(0).to(device))
            
            # Decode the latent representation to get the final images
            generated_images = vqvae.decode(xt)
            generated_images = torch.clamp(generated_images, -1., 1.).detach().cpu()
            generated_images = (generated_images + 1) / 2  # Normalize to [0, 1]
            
            all_samples.append(generated_images)
            
            # Save individual batch grid
            grid = make_grid(generated_images, nrow=min(8, actual_batch_size))
            img = torchvision.transforms.ToPILImage()(grid)
            img.save(os.path.join(output_dir, f'batch_{batch_idx+1}.png'))
    
    # Combine all samples and create a final grid
    all_samples = torch.cat(all_samples, dim=0)
    grid = make_grid(all_samples, nrow=int(num_samples**0.5))
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join(output_dir, 'all_samples.png'))
    print(f"Generated {num_samples} samples and saved them to {output_dir}")
    
    return all_samples

def full_generation_pipeline(vqvae, diffusion_model, scheduler, device, num_steps=1000):
    """
    Demonstrate the full generation pipeline with all diffusion steps.
    This is a more complete example showing the entire reverse process.
    """
    with torch.no_grad():
        # Start with random noise
        xt = torch.randn((1, 3, 4, 4)).to(device)
        
        # Iterate backward through timesteps
        for t in range(num_steps-1, -1, -1):
            print(f"Reverse diffusion step {num_steps-t}/{num_steps}", end="\r")
            
            # Create timestep tensor
            timestep = torch.tensor([t], device=device)
            
            # Predict noise
            noise_pred = diffusion_model(xt, timestep)
            
            # Sample x_{t-1} given x_t and predicted noise
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, timestep)
            
            # Optionally save intermediate steps
            if t % 100 == 0 or t == num_steps-1:
                # Decode the intermediate latent (for visualization)
                with torch.no_grad():
                    decoded = vqvae.decode(xt)
                    decoded = torch.clamp(decoded, -1., 1.).detach().cpu()
                    decoded = (decoded + 1) / 2
                    
                    img = torchvision.transforms.ToPILImage()(decoded[0])
                    os.makedirs("intermediate_steps", exist_ok=True)
                    img.save(f"intermediate_steps/step_{num_steps-t}.png")
        
        # Final decoding
        final_images = vqvae.decode(xt)
        final_images = torch.clamp(final_images, -1., 1.).detach().cpu()
        final_images = (final_images + 1) / 2
        
        return final_images

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # Load configuration
    config = get_config()
    
    # Initialize models
    vqvae = VQVAE().to(device)
    diffusion_model = UNet().to(device)  # Initialize your diffusion model (adjust as needed)
    
    # Load checkpoints
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
    diffusion_model.load_state_dict(torch.load(args.diffusion_checkpoint, map_location=device))
    
    # Put models in eval mode
    vqvae.eval()
    diffusion_model.eval()
    
    # Create noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195
    )
    
    # Generate samples
    samples = generate_samples(
        vqvae=vqvae,
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir
    )
    
    print("Sample generation complete!")

if __name__ == "__main__":
    main()
