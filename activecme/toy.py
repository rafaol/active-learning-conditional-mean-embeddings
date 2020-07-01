import torch
import torch.distributions
import gpytorch
from pyDOE import lhs


class GaussianConditional:
    """
    Gaussian state conditional distribution for toy experiment
    """
    def __init__(self, c_dim: int, s_dim: int, state_noise_sg: float):
        self.state_dim = s_dim
        self.transform_matrix = torch.randn(c_dim, s_dim)
        self.covariance_factor = torch.eye(s_dim)*state_noise_sg

    def conditional(self, control):
        return torch.distributions.MultivariateNormal(control @ self.transform_matrix,
                                                      scale_tril=self.covariance_factor)

    def __call__(self, control: torch.Tensor) -> torch.Tensor:
        return self.conditional(control).sample()


class RKHSProblem:
    """
    RKHS function for toy optimisation problem
    """
    def __init__(self, state_sampler: GaussianConditional, kernel: gpytorch.kernels.Kernel, n_features: int, n_dim: int,
                 obs_noise_sd=0.1):
        self.kernel = kernel
        self.kernel.eval()
        self.kernel.requires_grad_(False)
        points = torch.tensor(lhs(n_dim, samples=n_features, criterion='center'), dtype=torch.get_default_dtype())
        self.points = -torch.ones(n_dim) + 2*points
        K = kernel(self.points).evaluate()
        y = torch.distributions.MultivariateNormal(torch.zeros(n_features), K).sample()
        self.weights = torch.solve(y[:, None], K)[0].view(-1)
        self.state_sampler = state_sampler
        self.obs_noise_dist = torch.distributions.Normal(loc=torch.zeros([]), scale=obs_noise_sd)
        self.norm = (self.weights.t() @ K @ self.weights).sqrt()

    def objective(self, x):
        return self.kernel(x, self.points) @ self.weights

    def sample_state(self, control, n_samples=1):
        samples = torch.empty(n_samples, control.shape[0], self.state_sampler.state_dim)
        for i in range(n_samples):
            samples[i] = self.state_sampler(control)
        return samples

    def evaluate(self, control, n_samples=1):
        state_samples = self.sample_state(control, n_samples)
        f_samples = self.objective(state_samples)
        return state_samples, f_samples

    def observe(self, control):
        state_samples, f_samples = self.evaluate(control, n_samples=1)
        return state_samples[0], f_samples[0] + self.obs_noise_dist.sample(f_samples[0].size())


class GaussianEmbeddingKernel:
    """
    Kernel corresponding to the inner product between two Gaussian distribution embeddings with a RBF (a.k.a.
     squared-exponential) kernel
    """
    def __init__(self, lengthscale, input_stddev, dim=1):
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.kernel.base_kernel.lengthscale = (lengthscale**2 + input_stddev**2)**0.5
        self.kernel.outputscale = 1./(1+input_stddev**2/lengthscale**2)**0.5
        self.kernel.eval()
        self.kernel.requires_grad_(False)
        self.input_stddev = input_stddev
        self.lengthscale = lengthscale
        self.dim = dim

    def __call__(self, x, x_other=None):
        if x_other is None:
            x_other = x
        return self.kernel(x, x_other).evaluate()

    def norm(self):
        return torch.sqrt((1./(1+2*self.input_stddev**2/self.lengthscale**2)**0.5)**self.dim)


def find_optimum(prob, search_space, n_samples=100):
    mean_f = prob.evaluate(search_space, n_samples=n_samples)[1].mean(dim=0)
    opt_idx = mean_f.argmax()
    return search_space[opt_idx], mean_f[opt_idx]


def compute_regret(prob, controls, opt_f, n_samples=100):
    mean_f = prob.evaluate(controls, n_samples=n_samples)[1].mean(dim=0)
    return opt_f - mean_f
