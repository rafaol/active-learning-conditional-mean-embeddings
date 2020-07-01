"""
Acquisition functions
"""
import math
import torch
from activecme.cmemodel import CMEModel
from activecme.gp import GPModel
import abc
from typing import Union, Callable


class BaseUCB(abc.ABC):
    """
    Base class for UCB methods
    """
    def __init__(self, delta):
        self._beta_t = torch.zeros([])
        self.delta = delta

    @abc.abstractmethod
    def ucb_parameter(self, delta):
        pass

    def update(self):
        """
        Update UCB parameter according to confidence level.
        """
        self._beta_t = self.ucb_parameter(self.delta)

    @property
    def beta_t(self) -> torch.Tensor:
        """
        UCB parameter
        """
        return self._beta_t

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()


class PlainUCB(BaseUCB):
    """
    Gaussian process upper confidence bound (GP-UCB)
    """
    def __init__(self, gp_model: GPModel,
                 f_bound: Union[float, torch.Tensor],
                 sigma_out: Union[float, torch.Tensor] = 0.,
                 delta: Union[float, torch.Tensor] = 0.1):
        """
        Constructor.

        :param gp_model: a `GPModel` for the objective function
        :param f_bound: upper bound on the RKHS norm of the objective function
        :param sigma_out: sub-Gaussian parameter for the output noise
        :param delta: confidence level, i.e. UCB is valid with prob. >= 1-`delta` (Default: 0.1)
        """
        super().__init__(delta)
        self.gp_model = gp_model
        self.f_bound = f_bound
        self.sigma_out = sigma_out
        self._beta_t = self.ucb_parameter(delta)

    @property
    def name(self):
        return "GP-UCB"

    def ucb_parameter(self, delta):
        ig = self.gp_model.information_gain()
        lam = self.gp_model.likelihood.noise
        beta_kt = self.f_bound * (lam ** 0.5) + self.sigma_out * torch.sqrt(2 * (ig - math.log(delta)))
        return beta_kt

    def __call__(self, control: torch.Tensor) -> torch.Tensor:
        pred = self.gp_model(control)
        return pred.mean + self.beta_t * pred.variance.sqrt()/self.gp_model.likelihood.noise.sqrt()


class CMEUCB(PlainUCB):
    """
    Conditional mean embeddings upper confidence bound (CME-UCB)
    """
    def __init__(self, gp_model: GPModel, cme_model: CMEModel,
                 cme_bound: Union[float, torch.Tensor],
                 f_bound: Union[float, torch.Tensor],
                 sigma_in: Union[float, torch.Tensor] = 0.,
                 sigma_out: Union[float, torch.Tensor] = 0.,
                 delta: Union[float, torch.Tensor] = 0.1):
        """

        :param gp_model: a `GPModel` for the objective function
        :param cme_model: a `CMEModel` for the conditional mean embedding
        :param cme_bound: upper bound for the Hilbert-Schmidt norm of the CME operator
        :param f_bound: upper bound on the RKHS norm of the objective function
        :param sigma_in: sub-Gaussian parameter for the input noise effect on the kernel of the objective function
        :param sigma_out: sub-Gaussian parameter for the output noise
        :param delta: confidence level, i.e. UCB is valid with prob. >= 1-`delta` (Default: 0.1)
        """
        self.cme_model = cme_model
        self.cme_bound = cme_bound
        self.sigma_in = sigma_in
        super().__init__(gp_model, f_bound, sigma_out, delta)

    @property
    def name(self):
        return "CME-UCB"

    def c_parameter(self, delta):
        """
        Computes beta_c.

        :param delta: minimum probability
        :return:
        """
        ig = self.cme_model.information_gain()
        eta = self.cme_model.likelihood.noise
        beta_ct = self.cme_bound * (eta ** 0.5) + self.sigma_in * torch.sqrt(2 * (ig - math.log(delta)))
        return beta_ct.view([])

    def k_parameter(self, delta):
        """
        Computes beta_k.

        :param delta: minimum probability
        :return:
        """
        return super().ucb_parameter(delta)

    def ucb_parameter(self, delta):
        beta_c = self.c_parameter(delta/2)
        beta_k = self.k_parameter(delta/2)

        ig_k = self.gp_model.information_gain()
        k_factor = torch.sqrt(2*ig_k)

        beta_t = self.f_bound*beta_c + k_factor*beta_k

        return beta_t.view([])

    def f_mean(self, state):
        return self.gp_model(state).mean

    def __call__(self, control: torch.Tensor) -> torch.Tensor:
        pred = self.cme_model(control, self.f_mean)
        return pred.mean + self.beta_t * pred.variance.sqrt()/self.cme_model.likelihood.noise.sqrt()


class ImprovedCMEUCB(CMEUCB):
    """
    Improved CME-UCB.
    """
    def __init__(self, gp_model: GPModel, cme_model: CMEModel,
                 cme_bound: Union[float, torch.Tensor],
                 f_bound: Union[float, torch.Tensor],
                 sigma_in: Union[float, torch.Tensor] = 0.,
                 sigma_out: Union[float, torch.Tensor] = 0.,
                 delta: Union[float, torch.Tensor] = 0.1):
        """Constructor.

        :param gp_model: a `GPModel` for the objective function
        :param cme_model: a `CMEModel` for the conditional mean embedding
        :param cme_bound: upper bound for the Hilbert-Schmidt norm of the CME operator
        :param f_bound: upper bound on the RKHS norm of the objective function
        :param sigma_in: sub-Gaussian parameter for the input noise effect on the kernel of the objective function
        :param sigma_out: sub-Gaussian parameter for the output noise
        :param delta: confidence level, i.e. UCB is valid with prob. >= 1-delta (Default: 0.1)
        """
        super().__init__(gp_model, cme_model, cme_bound, f_bound, sigma_in, sigma_out, delta)

    @property
    def name(self):
        return "I-CME-UCB"

    def __call__(self, control: torch.Tensor) -> torch.Tensor:
        pred = self.cme_model(control, self.f_mean)

        if self.cme_model.U is not None:
            v = torch.cholesky_solve(self.cme_model.covar_module(self.cme_model.U, control).evaluate(),
                                     self.cme_model.chol_cov_data)
            gp_cov = self.gp_model(self.gp_model.X).covariance_matrix/self.gp_model.likelihood.noise
            variance = (v.t() @ gp_cov @ v).diagonal().view_as(pred.variance)
        else:
            variance = torch.zeros([])

        return pred.mean + self.c_parameter(self.delta/2)*pred.variance.sqrt()/self.cme_model.likelihood.noise.sqrt()\
                + self.k_parameter(self.delta/2)*variance.sqrt()


class QueryCMEUCB(BaseUCB):
    """
    Query-only CME-UCB for cases where the objective function is known, but the query distribution is not.
    """
    def __init__(self, cme_model: CMEModel,
                 cme_bound: Union[float, torch.Tensor],
                 sigma_in: Union[float, torch.Tensor] = 0.,
                 delta: Union[float, torch.Tensor] = 0.1):
        """
        Constructor.

        :param cme_model: a `CMEModel` for the conditional mean embedding
        :param cme_bound: upper bound for the Hilbert-Schmidt norm of the CME operator
        :param sigma_in: sub-Gaussian parameter for the input noise effect on the kernel of the objective function
        :param delta: confidence level, i.e. UCB is valid with prob. >= 1-delta (Default: 0.1)
        """
        super().__init__(delta)
        self.cme_model = cme_model
        self.cme_bound = cme_bound
        self.sigma_in = sigma_in
        self._beta_t = self.ucb_parameter(delta)

    @property
    def name(self):
        return "Q-CME-UCB"

    def ucb_parameter(self, delta):
        ig = self.cme_model.information_gain()
        eta = self.cme_model.likelihood.noise
        beta_ct = self.cme_bound * (eta ** 0.5) + self.sigma_in * torch.sqrt(2 * (ig - math.log(delta)))
        return beta_ct

    def __call__(self, control: torch.Tensor, fun: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Compute UCB on function value conditioned on the given control points.

        :param control: control points, a N-by-D array
        :param fun:
        :return:
        """
        pred = self.cme_model(control, fun)
        return pred.mean + self.beta_t * pred.variance.sqrt()/self.cme_model.likelihood.noise.sqrt()
