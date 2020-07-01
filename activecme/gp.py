"""
Implementation of GPyTorch's ExactGP with rank-1 Cholesky updates for applications in online learning.
"""
import gpytorch
import torch
from gpytorch import settings


class GPModel(gpytorch.models.ExactGP):
    """
    Gaussian process model
    """
    def __init__(self, covar_module: gpytorch.kernels.Kernel,
                 mean_module=gpytorch.means.ZeroMean(),
                 likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        """Constructor.

        :param covar_module: covariance function
        :type covar_module: gpytorch.kernels.Kernel
        :param mean_module: mean function (Default: `gpytorch.means.ZeroMean`)
        :param likelihood: likelihood model (Default: `gpytorch.likelihoods.GaussianLikelihood`)
        """
        super().__init__(None, None, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.cov_data = None
        self.chol_cov_data = None
        self.y_weights = None
        self.requires_grad_(False)

    def __getstate__(self):
        gpytorch_state_dict = self.state_dict()
        state_dict = {"data": (self.X, self.Y), "model": gpytorch_state_dict,
                      "covariance": self.covar_module.__class__, "mean": self.mean_module.__class__}
        return state_dict

    def __setstate__(self, state):
        self.__init__(covar_module=state["covariance"](), mean_module=state["mean"]())
        self.load_state_dict(state["model"])
        self.set_train_data(*state["data"])

    @property
    def X(self):
        if self.train_inputs is not None:
            return self.train_inputs[0]
        return None

    @property
    def Y(self):
        return self.train_targets

    def clear_data(self):
        self.train_inputs = None
        self.train_targets = None
        self.cov_data = None
        self.chol_cov_data = None
        self.y_weights = None

    def set_train_data(self, inputs=None, outputs=None, strict=False):
        super().set_train_data(inputs, outputs, strict=strict)
        self.cov_data = self.likelihood(self.forward(inputs), inputs).covariance_matrix
        self.chol_cov_data = torch.cholesky(self.cov_data)
        n = self.Y.shape[0]
        self.y_weights = torch.cholesky_solve((self.Y - self.mean_module(self.X)).view(n, -1), self.chol_cov_data)

    def update(self, x, y):
        if self.X is None:
            self.set_train_data(x, y)
        else:
            n = self.Y.shape[0]
            m = y.shape[0]
            cov_data_new = self.covar_module(self.X, x).evaluate()
            cov_new_new = self.likelihood(self.forward(x), x).covariance_matrix
            chol_data_new = torch.triangular_solve(cov_data_new, self.chol_cov_data, upper=False).solution
            chol_new_new = torch.cholesky(cov_new_new - chol_data_new.t() @ chol_data_new)
            self.chol_cov_data = torch.cat([torch.cat([self.chol_cov_data.detach(), torch.zeros(n, m)], dim=1),
                                            torch.cat([chol_data_new.t(), chol_new_new], dim=1)],
                                           dim=0)
            self.cov_data = torch.cat([torch.cat([self.cov_data.detach(), cov_data_new], dim=1),
                                       torch.cat([cov_data_new.t(), cov_new_new], dim=1)],
                                      dim=0)
            train_x, = self.train_inputs
            train_y = self.train_targets
            new_X = torch.cat([train_x, x])
            new_Y = torch.cat([train_y, y])
            self.y_weights = torch.cholesky_solve((new_Y - self.mean_module(new_X)).view(m+n, -1), self.chol_cov_data)
            super().set_train_data(new_X, new_Y, strict=False)

    def recalculate(self):
        """
        Recompute internals after updating hyper-parameters
        """
        self.set_train_data(self.X, self.Y)

    def information_gain(self):
        if self.X is not None:
            return torch.logdet(self.chol_cov_data/self.likelihood.noise.sqrt())
        return torch.zeros([])

    def get_noise_stddev(self):
        return self.likelihood.noise.sqrt().view([])

    def get_hyperparameters(self):
        return [p for p in self.parameters()]

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def __call__(self, x, full_cov=False):
        # Training mode
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not torch.equal(self.X, x):
                    raise RuntimeError("You must train on the training inputs!")
            return self.forward(x)

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_output = self.forward(x)
            if settings.debug().on():
                if not isinstance(full_output, gpytorch.distributions.MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            cov_data_query = self.covar_module(self.X, x).evaluate()
            prior_pred = self.forward(x)
            pred_mean = prior_pred.mean.view(-1, 1) + cov_data_query.t() @ self.y_weights
            cov_weights = torch.cholesky_solve(cov_data_query, self.chol_cov_data)

            if full_cov:
                pred_cov = prior_pred.covariance_matrix - cov_data_query.t() @ cov_weights
            else:  # Evaluates only diagonal (variances) as a diagonal lazy matrix
                diag_k = gpytorch.lazy.DiagLazyTensor(prior_pred.lazy_covariance_matrix.diag())
                pred_cov = diag_k.add_diag(-cov_data_query.t().matmul(cov_weights).diag())

        return gpytorch.distributions.MultivariateNormal(pred_mean.view_as(prior_pred.mean), pred_cov)
