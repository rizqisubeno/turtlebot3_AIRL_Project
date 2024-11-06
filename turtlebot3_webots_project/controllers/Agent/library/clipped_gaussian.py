import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def elementwise_gaussian_log_pdf(x, mean, var, ln_var):
    # log N(x|mean,var)
    #   = -0.5log(2pi) - 0.5log(var) - (x - mean)**2 / (2*var)
    return -0.5 * np.log(2 * np.pi) - 0.5 * ln_var - ((x - mean) ** 2) / (2 * var)

NPY_SQRT1_2 = 1 / (2 ** 0.5)

def _ndtr(a):
    """CDF of the standard normal distribution."""
    x = a * NPY_SQRT1_2
    z = torch.abs(x)
    half_erfc_z = 0.5 * torch.erfc(z)
    return torch.where(z < NPY_SQRT1_2,
                       0.5 + 0.5 * torch.erf(x),
                       torch.where(x > 0, 1.0 - half_erfc_z, half_erfc_z))

def _safe_log(x):
    """Logarithm function that won't backprop inf to input."""
    return torch.log(torch.where(x > 0, x, torch.tensor(1e-8, device=x.device)))

def _log_ndtr(x):
    """Log CDF of the standard normal distribution."""
    return torch.where(
        x > 6,
        -_ndtr(-x),
        torch.where(
            x > -14,
            _safe_log(_ndtr(x)),
            -0.5 * x * x - _safe_log(-x) - 0.5 * np.log(2 * np.pi)))

def _gaussian_log_cdf(x, mu, sigma):
    """Log CDF of a normal distribution."""
    return _log_ndtr((x - mu) / sigma)

def _gaussian_log_sf(x, mu, sigma):
    """Log SF (Survival Function) of a normal distribution."""
    return _log_ndtr(-(x - mu) / sigma)

class ClippedGaussian:
    """Clipped Gaussian distribution."""

    def __init__(self, mean, var, low, high):
        self.mean = mean
        self.var = var
        self.ln_var = torch.log(var)
        self.low = low.expand_as(mean)
        self.high = high.expand_as(mean)

    def sample(self):
        # Generate unclipped Gaussian samples
        unclipped = Normal(self.mean, self.var.sqrt()).sample()
        # Clip the values within the provided bounds
        # print(f"{unclipped.device=}")
        # print(f"{self.low.device=}")
        # print(f"{self.high.device=}")
        return torch.clamp(unclipped, self.low, self.high)

    def log_prob(self, x):
        # print(f"{type(x)}")
        # print(f"{type(self.mean)}")
        unclipped_elementwise_log_prob = elementwise_gaussian_log_pdf(
            x, self.mean, self.var, self.ln_var)
        # print(f"{unclipped_elementwise_log_prob=}")
        std = self.var.sqrt()
        low_log_prob = _gaussian_log_cdf(self.low, self.mean, std)
        high_log_prob = _gaussian_log_sf(self.high, self.mean, std)
        # print(f"{low_log_prob=}")
        # print(f"{high_log_prob=}")
        elementwise_log_prob = torch.where(
            x <= self.low,
            low_log_prob,
            torch.where(x >= self.high, high_log_prob, unclipped_elementwise_log_prob)
        )
        # print(f"{elementwise_log_prob=}")
        return elementwise_log_prob

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    def copy(self):
        return ClippedGaussian(self.mean.clone(), self.var.clone(), self.low.clone(), self.high.clone())
    
    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + torch.log(self.var.sqrt())
