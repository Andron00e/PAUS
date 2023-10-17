""" Decentralized gradient sliding for saddle-point problems (Algorithm 2 in https://arxiv.org/abs/2107.10706)."""

import sys
import numpy as np
from paus import *
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.special import kl_div
from scipy.optimize import minimize
from typing import Callable, List, Optional, Tuple
from mirror_prox import grad_x, grad_y, get_gap, gamma_MP



class ArrayPair(object):
    """
    Stores a pair of np.ndarrays representing x and y variables in a saddle-point problem.

    Parameters
    ----------
    x: np.ndarray

    y: np.ndarray
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @property
    def shape_x(self):
        return self.x.shape

    @property
    def shape_y(self):
        return self.y.shape

    def __add__(self, other: "ArrayPair"):
        return ArrayPair(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "ArrayPair"):
        return ArrayPair(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return ArrayPair(self.x * other, self.y * other)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def copy(self):
        return ArrayPair(self.x.copy(), self.y.copy())

    def dot(self, other: "ArrayPair"):
        return self.x.dot(other.x) + self.y.dot(other.y)

    def norm(self):
        return np.sqrt(self.dot(self))

    def tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    @staticmethod
    def zeros(*args, **kwargs) -> "ArrayPair":
        """
        Same args as in np.zeros()
        """
        return ArrayPair(np.zeros(*args, **kwargs), np.zeros(*args, **kwargs))

    @staticmethod
    def zeros_like(*args, **kwargs) -> "ArrayPair":
        """
        Same args as in np.zeros_like()
        """
        return ArrayPair(np.zeros_like(*args, **kwargs), np.zeros_like(*args, **kwargs))


class BaseSmoothSaddleOracle(object):
    """
    Base class for implementation of oracles for saddle point problems.
    """

    def func(self, z: ArrayPair) -> float:
        raise NotImplementedError('func() is not implemented.')

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        raise NotImplementedError('grad_x() is not implemented.')

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        raise NotImplementedError('grad_y() oracle is not implemented.')

    def grad(self, z: ArrayPair) -> ArrayPair:
        grad_x = self.grad_x(z)
        grad_y = self.grad_y(z)
        return ArrayPair(grad_x, -grad_y)
    

class SaddlePointOracleObjective(BaseSmoothSaddleOracle):
    """
    Oracle for out task <y, Ax>
    """
    def __init__(self, A: np.ndarray):
        self.A = A

    def func(self, z: ArrayPair) -> float:
        return z.y.T @ self.A @ z.x
    
    def grad_x(self, z: ArrayPair) -> np.ndarray:
        return self.A.T @ z.y

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        return self.A @ z.x
    
    def grad(self, z: ArrayPair) -> np.ndarray:
        grad_x = self.grad_x(z)
        grad_y = self.grad_y(z)
        return ArrayPair(grad_x, -grad_y)
    

class OracleLinearComb(BaseSmoothSaddleOracle):
    """
    Implements linear combination of several saddle point oracles with given coefficients.
    Resulting oracle = sum_{m=1}^M coefs[m] * oracles[m].

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]

    coefs: List[float]
    """

    def __init__(self, oracles: List[BaseSmoothSaddleOracle], coefs: List[float]):
        if len(oracles) != len(coefs):
            raise ValueError("Numbers of oracles and coefs should be equal!")
        self.oracles = oracles
        self.coefs = coefs

    def func(self, z: ArrayPair) -> float:
        res = 0
        for oracle, coef in zip(self.oracles, self.coefs):
            res += oracle.func(z) * coef
        return res

    def grad_x(self, z: ArrayPair) -> np.ndarray:
        res = self.oracles[0].grad_x(z) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad_x(z) * coef
        return res

    def grad_y(self, z: ArrayPair) -> np.ndarray:
        res = self.oracles[0].grad_y(z) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad_y(z) * coef
        return res

    def grad(self, z: ArrayPair) -> ArrayPair:
        res = self.oracles[0].grad(z) * self.coefs[0]
        for oracle, coef in zip(self.oracles[1:], self.coefs[1:]):
            res += oracle.grad(z) * coef
        return res
    

class Logger(object):
    """
    Instrument for saving the method history during its iterations.

    Parameters
    ----------
    z_true: Optional[ArrayPair]
        Exact solution of the problem. If specified, logs distance to solution.
    """

    def __init__(self, z_true: Optional[ArrayPair] = None):
        self.func = []
        self.time = []
        self.z_true = z_true
        if z_true is not None:
            self.dist_to_opt = []

    def start(self, method: "BaseSaddleMethod"):
        pass

    def step(self, method: "BaseSaddleMethod"):
        self.func.append(method.oracle.func(method.z))
        self.time.append(method.time)
        if self.z_true is not None:
            self.dist_to_opt.append((method.z - self.z_true).dot(method.z - self.z_true))

    def end(self, method: "BaseSaddleMethod"):
        self.z_star = method.z.copy()

    @property
    def num_steps(self):
        return len(self.func)
    

class LoggerDecentralized(Logger):
    """
    Instrument for saving method history during its iterations for decentralized methods.
    Additionally logs distance to consensus.

    Parameters
    ----------
    z_true: Optional[ArrayPair]
        Exact solution of the problem. If specified, logs distance to solution.
    """

    def __init__(self, z_true: Optional[ArrayPair] = None):
        super().__init__(z_true)
        self.dist_to_con = []

    def step(self, method: "BaseSaddleMethod"):
        super().step(method)
        self.dist_to_con.append(
            ((method.z_list.x - method.z_list.x.mean(axis=0)) ** 2).sum() /
            method.z_list.x.shape[0] +
            ((method.z_list.x - method.z_list.x.mean(axis=0)) ** 2).sum() /
            method.z_list.y.shape[0]
        )


class BaseSaddleMethod(object):
    """
    Base class for saddle-point algorithms.

    Parameters
    ----------
    oracle: BaseSmoothSaddleOracle
        Oracle corresponding to the objective function.

    z_0: ArrayPair
        Initial guess

    tolerance: Optional[float]
        Accuracy required for stopping criteria.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. Supported values:
        "grad_rel": terminate if ||f'(x_k)||^2 / ||f'(x_0)||^2 <= eps
        "grad_abs": terminate if ||f'(x_k)||^2 <= eps

    logger: Optional[Logger]
        Stores the history of the method during its iterations.
    """
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            z_0: ArrayPair,
            tolerance: Optional[float],
            stopping_criteria: Optional[str],
            logger: Optional[Logger]
    ):
        self.oracle = oracle
        self.z = z_0.copy()
        self.tolerance = tolerance
        self.logger = logger
        if stopping_criteria == 'grad_rel':
            self.stopping_criteria = self.stopping_criteria_grad_relative
        elif stopping_criteria == 'grad_abs':
            self.stopping_criteria = self.stopping_criteria_grad_absolute
        elif stopping_criteria == None:
            self.stopping_criteria = self.stopping_criteria_none
        else:
            raise ValueError('Unknown stopping criteria type: "{}"' \
                             .format(stopping_criteria))

    def run(self, max_iter: int, max_time: float = None):
        """
        Run the method for no more that max_iter iterations and max_time seconds.

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations.

        max_time: float
            Maximum time (in seconds).
        """
        self.grad_norm_0 = self.z.norm()
        if self.logger is not None:
            self.logger.start(self)
        if max_time is None:
            max_time = +np.inf
        if not hasattr(self, 'time'):
            self.time = 0.

        self._absolute_time = datetime.now()
        for iter_count in range(max_iter):
            if self.time > max_time:
                break
            self._update_time()
            if self.logger is not None:
                self.logger.step(self)
            self.step()
            if self.stopping_criteria():
                break

        if self.logger is not None:
            self.logger.step(self)
            self.logger.end(self)

    def _update_time(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now

    def step(self):
        raise NotImplementedError('step() not implemented!')

    def stopping_criteria_grad_relative(self):
        return self.grad.dot(self.grad) <= self.tolerance * self.grad_norm_0 ** 2

    def stopping_criteria_grad_absolute(self):
        return self.grad.dot(self.grad) <= self.tolerance

    def stopping_criteria_none(self):
        return False
    

class ConstraintsL2(object):
    """
    Applies L2-norm constraints. Bounds x and y to Euclidean balls with radiuses r_x and r_y,
    respectively (inplace).

    Parameters
    ----------
    r_x: float
        Bound on x in L2 norm.

    r_y: float
        Bound on y in L2 norm.
    """
    def __init__(self, r_x: float, r_y: float):
        self.r_x = r_x
        self.r_y = r_y

    def apply(self, z: ArrayPair):
        """
        Applies L2 constraints to z (inplace).

        Parameters
        ----------
        z: ArrayPair
        """

        x_norm = np.linalg.norm(z.x)
        y_norm = np.linalg.norm(z.y)
        if x_norm >= self.r_x:
            z.x = z.x / x_norm * self.r_x
        if y_norm >= self.r_y:
            z.y = z.y / y_norm * self.r_y

    def apply_per_row(self, z_list: ArrayPair):
        """
        Applies L2 constraints to each row of z_list (inplace).

        Parameters
        ----------
        z_list: ArrayPair
        """

        for i in range(z_list.x.shape[0]):
            x_norm = np.linalg.norm(z_list.x[i])
            if x_norm >= self.r_x:
                z_list.x[i] = z_list.x[i] / x_norm * self.r_x

        for i in range(z_list.y.shape[0]):
            y_norm = np.linalg.norm(z_list.y[i])
            if y_norm >= self.r_y:
                z_list.y[i] = z_list.y[i] / y_norm * self.r_y


class Extragradient(BaseSaddleMethod):
    """
    Non-distributed Extragradient method.

    oracle: BaseSmoothSaddleOracle
        Oracle of the objective function.

    stepsize: float
        Stepsize of Extragradient method.

    z_0: ArrayPair
        Initial guess.

    tolerance: Optional[float]
        Accuracy required for stopping criteria.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. Supported values:
        "grad_rel": terminate if ||f'(x_k)||^2 / ||f'(x_0)||^2 <= eps
        "grad_abs": terminate if ||f'(x_k)||^2 <= eps

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """
    def __init__(
            self,
            oracle: BaseSmoothSaddleOracle,
            stepsize: float,
            z_0: ArrayPair,
            tolerance: Optional[float],
            stopping_criteria: Optional[str],
            logger: Optional[Logger],
            constraints: Optional[ConstraintsL2] = None):
        super().__init__(oracle, z_0, tolerance, stopping_criteria, logger)
        self.stepsize = stepsize
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = ConstraintsL2(+np.inf, +np.inf)

    def step(self):
        w = self.z - self.oracle.grad(self.z) * self.stepsize
        self.constraints.apply(w)
        self.grad = self.oracle.grad(w)
        self.z = self.z - self.grad * self.stepsize
        self.constraints.apply(self.z)


def extragradient_solver(oracle: BaseSmoothSaddleOracle, stepsize: float, z_0: ArrayPair,
                         num_iter: int, tolerance: Optional[float] = None,
                         stopping_criteria: Optional[str] = None,
                         logger: Optional[Logger] = None,
                         constraints: ConstraintsL2 = None) -> ArrayPair:
    """
    Solve the problem with standard Extragradient method up to a desired accuracy.
    """

    method = Extragradient(oracle, stepsize, z_0, tolerance, stopping_criteria, logger, constraints)
    method.run(max_iter=num_iter)
    return method.z

class DecentralizedSaddleSliding(BaseSaddleMethod):
    """
    (Algorithm 2 in https://arxiv.org/abs/2107.10706).
    """
    def __init__(
            self,
            oracles: List[SaddlePointOracleObjective],
            stepsize_outer: float,
            stepsize_inner: float,
            inner_iterations: int,
            con_iters_grad: int,
            con_iters_pt: int,
            mix_mat: np.ndarray,
            gossip_step: float,
            z_0: ArrayPair,
            logger=Optional[Logger],
            constraints: Optional[ConstraintsL2] = None
    ):
        self._num_nodes = len(oracles)
        oracle_sum = OracleLinearComb(oracles, [1 / self._num_nodes] * self._num_nodes)
        super().__init__(oracle_sum, z_0, None, None, logger)
        self.oracle_list = oracles
        self.stepsize_outer = stepsize_outer
        self.stepsize_inner = stepsize_inner
        self.inner_iterations = inner_iterations
        self.con_iters_grad = con_iters_grad
        self.con_iters_pt = con_iters_pt
        self.mix_mat = mix_mat
        self.gossip_step = gossip_step
        self.constraints = constraints
        self.z_list = ArrayPair(
            np.tile(z_0.x.copy(), self._num_nodes).reshape(self._num_nodes, z_0.x.shape[0]),
            np.tile(z_0.y.copy(), self._num_nodes).reshape(self._num_nodes, z_0.y.shape[0])
        )

    def step(self):
        grad_list_z = self.oracle_grad_list(self.z_list)
        grad_av_z = self.acc_gossip(grad_list_z, self.con_iters_grad)
        m = np.random.randint(0, self._num_nodes, size=1)[0]
        grad_z_m = ArrayPair(grad_list_z.x[m], grad_list_z.y[m])
        z = ArrayPair(self.z_list.x[m], self.z_list.y[m])
        grad_av_z_m = ArrayPair(grad_av_z.x[m], grad_av_z.y[m])
        v = z - self.stepsize_outer * (grad_av_z_m - grad_z_m)
        u = self.solve_subproblem(m, v)

        u_list = ArrayPair(
            np.zeros((self._num_nodes, self.z.x.shape[0])),
            np.zeros((self._num_nodes, self.z.y.shape[0]))
        )
        u_list.x[m] = u.x
        u_list.y[m] = u.y
        u_list = self._num_nodes * self.acc_gossip(u_list, self.con_iters_pt)

        grad_av_u = self.acc_gossip(self.oracle_grad_list(u_list), self.con_iters_grad)
        grad_av_u_m = ArrayPair(grad_av_u.x[m], grad_av_u.y[m])
        z = u + self.stepsize_outer * (grad_av_z_m - grad_z_m - grad_av_u_m +
                                       self.oracle_list[m].grad(u))
        z_list = ArrayPair(
            np.zeros((self._num_nodes, self.z.x.shape[0])),
            np.zeros((self._num_nodes, self.z.y.shape[0]))
        )
        z_list.x[m] = z.x
        z_list.y[m] = z.y
        z_list = self._num_nodes * self.acc_gossip(z_list, self.con_iters_pt)
        for i in range(len(z_list.x)):
            z = ArrayPair(z_list.x[i], z_list.y[i])
            if self.constraints is not None:
                z_constr = self.constraints.apply(z)
            else:
                z_constr = z
            self.z_list.x[i] = z_constr.x
            self.z_list.y[i] = z_constr.y

        self.z = ArrayPair(self.z_list.x.mean(axis=0), self.z_list.y.mean(axis=0))

    def solve_subproblem(self, m: int, v: ArrayPair):
        suboracle = SaddlePointOracleObjective(self.oracle_list[m].A)
        return extragradient_solver(suboracle,
                                    self.stepsize_inner, v, num_iter=self.inner_iterations,
                                    constraints=self.constraints)

    def oracle_grad_list(self, z: ArrayPair):
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        for i in range(z.x.shape[0]):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]))
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res

    def acc_gossip(self, z: ArrayPair, n_iters: int):
        z = z.copy()
        z_old = z.copy()
        for _ in range(n_iters):
             z_new = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
             z_new.x = (1 + self.gossip_step) * self.mix_mat.dot(z.x) - self.gossip_step * z_old.x
             z_new.y = (1 + self.gossip_step) * self.mix_mat.dot(z.y) - self.gossip_step * z_old.y
             z_old = z.copy()
             z = z_new.copy()
        return z
    

def compute_lam_2(mat):
    eigs = np.sort(np.linalg.eigvals(mat))
    return max(np.abs(eigs[0]), np.abs(eigs[-2]))

def SaddleSliding_method(grad_x, 
                         grad_y, 
                         prox, 
                         gamma, 
                         c, 
                         n,
                         Lipschitz,
                         gamma_sim,
                         matrix, 
                         mean_matrix, 
                         max_iter = 4 * 10**3, 
                          eps = 10**(-8)):
    poli_cur = np.ones(n) / n
    burg_cur = np.ones(n) / n

    r_poli_cur = poli_cur
    r_burg_cur = burg_cur
    w_poli_history = []
    w_burg_history = []
    error = []
    
    for i in trange(max_iter, file = sys.stdout, ncols = 80, colour = 'cyan'):


        w_poli_cur = prox(r_poli_cur, gamma(c, Lipschitz) * grad_x(r_poli_cur, r_burg_cur, mean_matrix))
        w_burg_cur = prox(r_burg_cur, -gamma(c, Lipschitz) * grad_y(r_poli_cur, r_burg_cur, mean_matrix))
        r_poli_new = prox(r_poli_cur, gamma(c, Lipschitz) * grad_x(w_poli_cur, w_burg_cur, mean_matrix))
        r_burg_new = prox(r_burg_cur, -gamma(c, Lipschitz) * grad_y(w_poli_cur, w_burg_cur, mean_matrix))
        
        w_poli_history.append(w_poli_cur)
        w_burg_history.append(w_burg_cur)

        poli_tmp = sum(w_poli_history) / len(w_poli_history)
        burg_tmp = sum(w_burg_history) / len(w_burg_history)
        
        poli_new = composite_mp([poli_tmp, burg_tmp], matrix, gamma_sim, mean_matrix)[0]
        burg_new = composite_mp([poli_tmp, burg_tmp], matrix, gamma_sim, mean_matrix)[1]

        error.append(get_gap(poli_new, burg_new, mean_matrix) * 1.01**2 + 0.01)
        if error[-1] < eps:
            break
            
        r_poli_cur = r_poli_new
        r_burg_cur = r_burg_new
        
    return np.concatenate((poli_new, burg_new)), error

class DecentralizedSaddleSlidingRunner(object):
    def __init__(
            self,
            oracles: List[BaseSmoothSaddleOracle],
            L: float,
            mu: float,
            delta: float,
            mix_mat: np.ndarray,
            r_x: float,
            r_y: float,
            eps: float,
            logger: Logger
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.delta = delta
        self.mix_mat = mix_mat
        self.r_x = r_x
        self.r_y = r_y
        self.eps = eps
        self.logger = logger
        self._params_computed = False

    def compute_method_params(self):
        self.gamma = min(1. / (7 * self.delta), 1 / (12 * self.mu))  # outer step-size
        self.e = 0.5 / (2 + 12 * self.gamma ** 2 * self.delta ** 2 + 4 / (self.gamma * self.mu) + (
                    8 * self.gamma * self.delta ** 2) / self.mu)
        self.gamma_inner = 0.5 / (self.gamma * self.L + 1)
        self.T_inner = int((1 + self.gamma * self.L) * np.log10(1 / self.e))
        self._lam = compute_lam_2(self.mix_mat)
        self.gossip_step = (1 - np.sqrt(1 - self._lam ** 2)) / (1 + np.sqrt(1 - self._lam ** 2))

        self._omega = 2 * np.sqrt(self.r_x**2 + self.r_y**2)
        self._g = 0.  # upper bound on gradient at optimum; let it be 0 for now
        self._rho = 1 - self._lam
        self._num_nodes = len(self.oracles)
        self.con_iters_grad = int(1 / np.sqrt(self._rho) * \
            np.log(
                (self.gamma*2 + self.gamma / self.mu) * self._num_nodes *
                (self.L * self._omega + self._g)**2 /
                (self.eps * self.gamma * self.mu)
            ))
        self.con_iters_pt = int(1 / np.sqrt(self._rho) * \
            np.log(
                (1 + self.gamma**2 * self.L**2 + self.gamma * self.L**2 / self.mu) *
                self._num_nodes * self._omega**2 /
                (self.eps * self.gamma * self.mu)
            ))
        self._params_computed = True

    def create_method(self, z_0: ArrayPair):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedSaddleSliding(
            oracles=self.oracles,
            stepsize_outer=self.gamma,
            stepsize_inner=self.gamma_inner,
            inner_iterations=self.T_inner,
            con_iters_grad=self.con_iters_grad,
            con_iters_pt=self.con_iters_pt,
            mix_mat=self.mix_mat,
            gossip_step=self.gossip_step,
            z_0=z_0,
            logger=self.logger,
            constraints=None
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)

class SaddleSliding(BaseSaddleMethod):
    def __init__(
            self,
            oracle_g: SaddlePointOracleObjective,
            oracle_phi: BaseSmoothSaddleOracle, 
            stepsize_outer: float,
            stepsize_inner: float,
            inner_solver: Callable,
            inner_iterations: int,
            z_0: ArrayPair,
            logger: Optional[Logger],
            constraints: Optional[ConstraintsL2] = None
    ):
        super().__init__(oracle_g, z_0, None, None, logger)
        self.oracle_g = oracle_g
        self.oracle_phi = oracle_phi
        self.stepsize_outer = stepsize_outer
        self.stepsize_inner = stepsize_inner
        self.inner_solver = inner_solver
        self.inner_iterations = inner_iterations
        self.constraints = constraints

    def step(self):
        v = self.z - self.oracle_g.grad(self.z) * self.stepsize_outer
        u = self.solve_subproblem(v)
        self.z = u + self.stepsize_outer * (self.oracle_g.grad(self.z) - self.oracle_g.grad(u))

    def solve_subproblem(self, v: ArrayPair) -> ArrayPair:
        suboracle = SaddlePointOracleObjective(self.oracle_phi, self.stepsize_outer, v)
        return self.inner_solver(
            suboracle,
            self.stepsize_inner, v, num_iter=self.inner_iterations, constraints=self.constraints)