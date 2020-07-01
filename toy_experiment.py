"""
Toy experiment
"""

import sys
from tqdm import trange
import gpytorch
import torch
import torch.distributions
from activecme import toy
from activecme.cmemodel import CMEModel
from activecme.gp import GPModel
from activecme.af import ImprovedCMEUCB, CMEUCB, PlainUCB

from pyDOE import lhs


class BoundError(AssertionError):
    def __init__(self, *args):
        super().__init__(*args)


def setup_cme_ucb(control_kernel: gpytorch.kernels.Kernel, state_kernel: gpytorch.kernels.Kernel, state_dim: int,
                  state_dist: toy.GaussianConditional, obs_noise_sd: float, search_space: torch.Tensor,
                  f_norm_bound: torch.Tensor, delta: float, improved_ucb=False, max_cme_norm=10):
    # CME model setup
    cme_model = CMEModel(control_kernel)
    cme_model.eval()
    eta = 0.01
    cme_model.likelihood.noise = eta

    # GP models setup:
    gp_model = GPModel(state_kernel)
    gp_model.eval()
    lam = 1.
    gp_model.likelihood.noise = lam

    # UCB setup
    full_conditional = state_dist.conditional(search_space)
    state_noise_sd = full_conditional.variance.max().sqrt()

    # CME norm computation
    embedding_kernel = toy.GaussianEmbeddingKernel(state_kernel.lengthscale, state_noise_sd, dim=state_dim)
    embedding_matrix = embedding_kernel(full_conditional.mean)
    c_mat = control_kernel(search_space).evaluate()
    eps = torch.symeig(c_mat).eigenvalues.min().abs() + 1e-7
    c_chol_inv = torch.cholesky(c_mat + eps * torch.eye(*c_mat.shape)).inverse()

    cme_norm_bound = torch.symeig(c_chol_inv @ embedding_matrix @ c_chol_inv.t()).eigenvalues.max().sqrt()
    if cme_norm_bound > max_cme_norm:
        raise BoundError("CME too large (norm: {} > {})".format(cme_norm_bound.item(), max_cme_norm))

    sg_sigma_in = 2
    sg_sigma_out = obs_noise_sd

    if improved_ucb:
        af = ImprovedCMEUCB(gp_model, cme_model, cme_norm_bound, f_norm_bound, sg_sigma_in, sg_sigma_out, delta)
    else:
        af = CMEUCB(gp_model, cme_model, cme_norm_bound, f_norm_bound, sg_sigma_in, sg_sigma_out, delta)

    return af


def setup_gp_ucb(control_kernel, obs_noise_sd: float, f_norm_bound: torch.Tensor, delta: float):
    # Baseline GP setup
    plain_gp_kernel = gpytorch.kernels.RBFKernel()
    plain_gp_kernel.lengthscale = control_kernel.lengthscale
    plain_gp_model = GPModel(plain_gp_kernel)
    plain_gp_model.likelihood.noise = 1
    plain_gp_model.eval()

    # UCB configuration
    sg_sigma_in = 2
    sg_sigma_out = obs_noise_sd

    plain_af = PlainUCB(plain_gp_model, f_norm_bound,
                        sigma_out=(sg_sigma_out ** 2 + f_norm_bound ** 2 * sg_sigma_in ** 2) ** 0.5, delta=delta)

    return plain_af


def setup_problem(search_space: torch.Tensor, state_kernel: gpytorch.kernels.Kernel, c_dim: int, state_dim: int,
                  s_noise_sg: float, y_noise_sd: float, n_bases: int):
    state_dist = toy.GaussianConditional(c_dim, state_dim, s_noise_sg)

    scaled_kernel = gpytorch.kernels.ScaleKernel(state_kernel)
    scaled_kernel.outputscale = 5 * c_dim ** 2

    problem = toy.RKHSProblem(state_dist, scaled_kernel, n_features=n_bases, n_dim=state_dim, obs_noise_sd=y_noise_sd)
    f_values = torch.stack([problem.objective(state_dist(search_space)) for _ in range(100)]).mean(dim=0)
    sys.stdout.write("RKHS norm: {}\nf in [{:.3f}, {:.3f}]\n".format(problem.norm, f_values.min().item(),
                                                                     f_values.max().item()))

    sys.stdout.flush()
    sys.stderr.flush()

    control_kernel = gpytorch.kernels.MaternKernel()
    control_kernel.lengthscale = 0.1

    # UCB setup
    delta = 0.2
    f_norm_bound = problem.norm

    cme_ucb = setup_cme_ucb(control_kernel, state_kernel, state_dim, state_dist, y_noise_sd, search_space, f_norm_bound,
                            delta, improved_ucb=False)

    improved_cme_ucb = setup_cme_ucb(control_kernel, state_kernel, state_dim, state_dist, y_noise_sd, search_space,
                                     f_norm_bound, delta, improved_ucb=True)

    gp_ucb = setup_gp_ucb(control_kernel, y_noise_sd, f_norm_bound, delta)

    return gp_ucb, cme_ucb, improved_cme_ucb, problem


def loop(acquisition_functions, search_space, optimisation_problem, n_iterations: int = 100):
    # BO Loop
    n_af = len(acquisition_functions)

    pbar = trange(n_iterations)
    for t in pbar:
        y_values = torch.zeros(n_af)
        for i, af in enumerate(acquisition_functions):
            af_values = af(search_space)
            af_idx = af_values.argmax()

            u_t = search_space[af_idx].view(1, -1)
            x_t, y_t = optimisation_problem.observe(u_t)
            y_values[i] = y_t

            if isinstance(af, CMEUCB):
                af.cme_model.update(u_t, x_t)
            af.gp_model.update(x_t, y_t)

            af.update()

        pbar.set_postfix({af.name: y.item() for af, y in zip(acquisition_functions, y_values)})

    pbar.close()
    sys.stdout.flush()
    sys.stderr.flush()


def run_bo(search_space, s_kernel, ss_dim, s_noise_sg, obs_noise_sg, n_bases, num_iterations):
    baseline_ucb, cme_ucb, improved_cme_ucb, optimisation_problem = setup_problem(search_space, s_kernel,
                                                                                  search_space.shape[-1],
                                                                                  ss_dim, s_noise_sg,
                                                                                  obs_noise_sg, n_bases)

    n_samples = 200
    opt_control, opt_f_mean = toy.find_optimum(optimisation_problem, search_space, n_samples=n_samples)
    sys.stdout.write("Optimal control: {}\nOptimum value: {:.3f}\n".format(opt_control.numpy(), opt_f_mean.item()))
    sys.stdout.flush()
    sys.stderr.flush()

    methods = [baseline_ucb, cme_ucb, improved_cme_ucb]

    loop(methods, search_space, optimisation_problem, num_iterations)

    regret = {}
    for method in methods:
        if isinstance(method, CMEUCB):
            queries = method.cme_model.U
        else:
            queries = method.gp_model.X
        regret[method.name] = toy.compute_regret(optimisation_problem, queries, opt_f_mean, n_samples=n_samples)

    return regret


def main():
    state_dim = 1
    control_dim = 1
    n_controls = 100  # number of elements in control space

    control_space = torch.tensor(lhs(control_dim, samples=n_controls, criterion="center"),
                                 dtype=torch.get_default_dtype())  # create control set via latin hyper-cube sampling

    # RKHS setup
    state_kernel = gpytorch.kernels.RBFKernel()
    state_kernel.lengthscale = 0.05

    # Objective setup
    state_noise_sd = 0.01
    obs_noise_sd = 0.05
    n_features = 40

    # Run BO
    n_iterations = 500
    n_repeats = 10

    regrets = {}
    rep = 0
    while rep < n_repeats:
        sys.stdout.write("==== Run {} of {} ====\n".format(rep+1, n_repeats))
        try:
            regret = run_bo(control_space, state_kernel, state_dim, state_noise_sd, obs_noise_sd, n_features,
                            n_iterations)
        except BoundError as err:
            sys.stderr.write("Invalid setting: {}\nTrying again...\n".format(err))
            continue
        except RuntimeError as err:
            sys.stderr.write("Cholesky decomposition error: {}\nTrying again...\n".format(err))
            continue
        if len(regrets) == 0:
            for name, r in regret.items():
                regrets[name] = [r]
        else:
            for name, r in regret.items():
                regrets[name] += [r]
        rep += 1

    for name, r_list in regrets.items():
        regret = torch.stack(r_list, dim=0)
        sys.stdout.write("{} cumulative regret: {} +/- {}\n".format(name,
                                                                    regret.sum(dim=1).mean(),
                                                                    regret.sum(dim=1).std())
                         )
        torch.save(regret, "{}-regret.pth".format(name.lower()))


if __name__ == "__main__":
    main()
    sys.stdout.write("Done\n")
