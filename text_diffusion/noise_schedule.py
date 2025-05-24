import numpy as np


class LinearNoiseSchedule:
    """Implements a linear noise schedule for diffusion models.

    This class computes and stores the beta, alpha, and alpha_bar
    values that control how noise is added during the forward diffusion process.
    """

    def __init__(self, n_timesteps=10_000, beta_start=0.0001, beta_end=0.02):
        self.n_timesteps = n_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self._compute_beta_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)

    def _compute_beta_schedule(self):
        return np.linspace(self.beta_start, self.beta_end, self.n_timesteps)

    def add_noise(self, x0, noise, t):
        """Add noise to clean data x0 at timestep t."""
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
