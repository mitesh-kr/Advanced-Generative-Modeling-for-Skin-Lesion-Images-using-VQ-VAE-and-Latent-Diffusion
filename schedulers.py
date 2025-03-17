import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Calculate betas linearly
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Calculate alphas
        self.alphas = 1.0 - (self.betas - beta_start) / (beta_end - beta_start)

        # Compute cumulative product of alphas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)

        # Calculate square roots
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        """
        Add noise to the original tensor according to the diffusion schedule.
        
        Args:
            original: Original image tensor
            noise: Noise tensor with same shape as original
            t: Timestep tensor with shape [batch_size]
            
        Returns:
            Noisy tensor at timestep t
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        # Reshape tensors
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size, 1, 1, 1)

        # Apply forward process equation
        return (sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise)

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        Sample from p(x_{t-1} | x_t) using the predicted noise.
        
        Args:
            xt: Tensor at timestep t
            noise_pred: Predicted noise by model
            t: Current timestep
            
        Returns:
            Tuple of (sample for timestep t-1, predicted x0)
        """
        # Calculate mean and variance
        mean = xt - ((self.betas[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t])
        mean /= torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, ((xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) /
                          torch.sqrt(self.alpha_cum_prod[t]))
        else:
            variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t]) * self.betas[t]
            sigma = torch.sqrt(variance)
            z = torch.randn_like(xt)
            return mean + sigma * z, mean / torch.sqrt(1 - self.alpha_cum_prod[t - 1])
