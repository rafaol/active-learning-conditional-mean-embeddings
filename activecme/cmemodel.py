"""
Conditional mean embeddings model
"""
import gpytorch
from gpytorch.constraints import GreaterThan
import torch
from gpytorch import settings
from typing import Callable


class CMEModel(gpytorch.models.GP):
    """
    Conditional mean embeddings model

    Implementation of a GP-based model for conditional mean embeddings with rank-1 Cholesky updates.
    """
    def __init__(self, covar_module: gpytorch.kernels.Kernel, mean_module=None, likelihood=None, use_songs=False):
        """Constructor.

        :param covar_module: GPyTorch kernel module to be used as covariance function
        :param mean_module: mean function. If `None`, defaults to GPyTorch's ZeroMean
        :param likelihood: a Gaussian likelihood module. If `None` defaults to GPyTorch's GaussianLikelihood
        :param use_songs: whether or not to use Le Song's formulation for the CME estimator, regularising by the
        number of data points.
        """
        super().__init__()
        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module
        self.covar_module = covar_module
        self.cov_data = None
        self.chol_cov_data = None
        self.y_weights = None
        if likelihood is None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1-10))
        else:
            self.likelihood = likelihood
        self.train_inputs = None
        self.train_targets = None
        self.requires_grad_(False)
        self.use_songs = use_songs

    def __getstate__(self):
        gpytorch_state_dict = self.state_dict()
        state_dict = {"data": (self.U, self.X), "model": gpytorch_state_dict,
                      "covariance": self.covar_module.__class__, "mean": self.mean_module.__class__}
        return state_dict

    def __setstate__(self, state):
        self.__init__(covar_module=state["covariance"](), mean_module=state["mean"]())
        self.load_state_dict(state["model"])
        self.set_train_data(*state["data"])

    @property
    def U(self):
        return self.train_inputs

    @property
    def X(self):
        return self.train_targets

    def clear_data(self):
        self.train_inputs = None
        self.train_targets = None
        self.cov_data = None
        self.chol_cov_data = None
        self.y_weights = None

    def set_train_data(self, inputs=None, outputs=None):
        if inputs is None:
            self.clear_data()
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(outputs, torch.Tensor)
        assert inputs.shape[0] == outputs.shape[0]
        if self.use_songs:
            n = inputs.shape[0]
            self.cov_data = self.covar_module(inputs).evaluate() + n * self.likelihood.noise * torch.eye(
                inputs.shape[0])
        else:
            self.cov_data = self.covar_module(inputs).evaluate() + self.likelihood.noise * torch.eye(inputs.shape[0])
        self.chol_cov_data = torch.cholesky(self.cov_data)
        self.train_inputs = inputs
        self.train_targets = outputs

    def update(self, u, x):
        if self.X is None:
            self.set_train_data(u, x)
        else:
            new_U = torch.cat([self.U, u])
            new_X = torch.cat([self.X, x])
            if self.use_songs:
                self.set_train_data(new_U, new_X)
            else:
                n = self.U.shape[0]
                m = u.shape[0]
                cov_data_new = self.covar_module(self.U, u).evaluate()
                cov_new_new = self.covar_module(u).evaluate() + self.likelihood.noise * torch.eye(m)
                chol_data_new = torch.triangular_solve(cov_data_new, self.chol_cov_data, upper=False).solution
                chol_new_new = torch.cholesky(cov_new_new - chol_data_new.t() @ chol_data_new)
                self.chol_cov_data = torch.cat([torch.cat([self.chol_cov_data.detach(), torch.zeros(n, m)], dim=1),
                                                torch.cat([chol_data_new.t(), chol_new_new], dim=1)],
                                               dim=0)
                self.cov_data = torch.cat([torch.cat([self.cov_data.detach(), cov_data_new], dim=1),
                                           torch.cat([cov_data_new.t(), cov_new_new], dim=1)],
                                          dim=0)
                self.train_inputs = new_U
                self.train_targets = new_X

    def information_gain(self):
        if self.X is not None:
            if self.use_songs:
                n_data = self.X.shape[0]
            else:
                n_data = 1
            return torch.logdet(self.chol_cov_data / (n_data * self.likelihood.noise).sqrt())
        return torch.zeros([])

    def get_noise_stddev(self):
        return self.likelihood.noise.sqrt()[0]

    def get_hyperparameters(self):
        return [p for p in self.parameters()]

    def recalculate(self):
        """
        Recompute internals after updating hyper-parameters
        """
        self.set_train_data(self.U, self.X)

    def estimate_norm(self, x_kernel: gpytorch.kernels.Kernel) -> torch.Tensor:
        c_tf = torch.cholesky_solve(self.covar_module(self.U).evaluate(), self.chol_cov_data)
        k_tf = torch.cholesky_solve(x_kernel(self.X).evaluate(), self.chol_cov_data)
        norm = torch.trace(k_tf @ c_tf).sqrt()
        return norm

    def forward(self, u):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(u),
                                                         covariance_matrix=self.covar_module(u))

    def __call__(self, u: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], full_cov: bool = False):
        """Conditional mean prediction.

        :param u: control points (N-by-D array)
        :param f: callable implementing function whose expected value we want to compute.
        :param full_cov: whether or not to compute the full predictive covariance matrix, according to control kernel
        :return: a multivariate normal with the predictions
        """
        # Training mode
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                    )
            if settings.debug.on():
                if not torch.equal(self.U, u):
                    raise RuntimeError("You must train on the training inputs!")
            return self.forward(u)

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_output = self.forward(u)
            if settings.debug().on():
                if not isinstance(full_output, gpytorch.distributions.MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            cov_data_query = self.covar_module(self.U, u)
            prior_pred = self.forward(u)
            f_weights = torch.cholesky_solve(f(self.X).view(-1, 1), self.chol_cov_data)
            pred_mean = prior_pred.mean.view(-1, 1) + cov_data_query.t() @ f_weights
            cov_weights = torch.cholesky_solve(cov_data_query.evaluate(), self.chol_cov_data)

            if full_cov:
                pred_cov = prior_pred.covariance_matrix - cov_data_query.t().evaluate() @ cov_weights
            else:  # Evaluates only diagonal (variances) as a diagonal lazy matrix
                diag_k = gpytorch.lazy.DiagLazyTensor(prior_pred.lazy_covariance_matrix.diag())
                pred_cov = diag_k.add_diag(-cov_data_query.t().matmul(cov_weights).diag())

        return gpytorch.distributions.MultivariateNormal(pred_mean.view(-1), pred_cov)
