import sys

import numpy as _np

import netket as _nk
from .operator import local_values as _local_values

from netket.stats import (
    statistics as _statistics,
    covariance_sv as _covariance_sv,
    subtract_mean as _subtract_mean,
    mean as _mean,
)

from mpi4py import MPI as _MPI


def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


def make_optimizer_fn(arg, ma):
    """
    Utility function to create the optimizer step function for VMC drivers.

    It currently supports three kinds of inputs:

    1. A NetKet optimizer, i.e., a subclass of `netket.optimizer.Optimizer`.

    2. A 3-tuple (init, update, get) of optimizer functions as used by the JAX
       optimizer submodule (jax.experimental.optimizers).

       The update step p0 -> p1 with bare step dp is computed as
            x0 = init(p0)
            x1 = update(i, dp, x1)
            p1 = get(x1)

    3. A single function update with signature p1 = update(i, dp, p0) returning the
       updated parameter value.
    """
    if isinstance(arg, tuple) and len(arg) == 3:
        init, update, get = arg

        def optimize_fn(i, grad, p):
            x0 = init(p)
            x1 = update(i, grad, x0)
            return get(x1)

        desc = "JAX-like optimizer"
        return optimize_fn, desc

    elif issubclass(type(arg), _nk.optimizer.Optimizer):

        arg.init(ma.n_par, ma.is_holomorphic)

        def optimize_fn(_, grad, p):
            arg.update(grad, p)
            return p

        desc = info(arg)
        return optimize_fn, desc

    elif callable(arg):
        import inspect

        sig = inspect.signature(arg)
        if not len(sig.parameters) == 3:
            raise ValueError(
                "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
                + " or callable f(i, grad, p); got callable with signature {}".format(
                    sig
                )
            )
        desc = "{}{}".format(arg.__name__, sig)
        return arg, desc
    else:
        raise ValueError(
            "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
            + " or callable f(i, grad, p); got {}".format(arg)
        )


class VmcExact(object):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, hamiltonian, machine, optimizer, sr=None
    ):
        self._mynode = _MPI.COMM_WORLD.Get_rank()
        self._nnodes = _MPI.COMM_WORLD.Get_size()

        self._ham = hamiltonian
        self._hilbert = machine.hilbert
        self._machine = machine
        self._sr = sr
        self._stats = None

        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

        self._npar = self._machine.n_par

        self._dim = self._hilbert.n_states

        if self._mynode != (self._nnodes - 1):
            self._states_per_node = (self._dim // self._nnodes)
        else:
            self._states_per_node = self._dim - \
                (self._dim // self._nnodes) * (self._nnodes - 1)

        self._states_start = self._mynode * (self._dim // self._nnodes)
        self._states_end = self._states_start + self._states_per_node

        self._states = _np.array(
            [self._hilbert.number_to_state(i) for i in range(self._states_start, self._states_end)])

        self._wf = _np.ndarray(self._states_per_node, dtype=_np.float64)
        self._prob = _np.ndarray(self._states_per_node, dtype=_np.float64)
        self._der_logs = _np.ndarray(
            (self._states_per_node, self._npar), dtype=_np.complex128
        )

        self._grads = _np.empty(
            (self._states_per_node, self._npar), dtype=_np.complex128
        )

        self._obs = {}

        self.step_count = 0

        if self._mynode == 0:
            print("# Hilbert Size = ", self._dim)
            print("# Running on", self._nnodes, "processes")

    def advance(self, n_steps=1):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        for _ in range(n_steps):
            # Generate samples and store them
            for i, state in enumerate(self._states):
                self._wf[i] = _np.exp(_np.real(self._machine.log_val(state)))

            # MPI
            weight = _np.linalg.norm(self._wf, keepdims=True)**2
            _MPI.COMM_WORLD.Allreduce(_MPI.IN_PLACE, weight, op=_MPI.SUM)

            self._wf /= _np.sqrt(weight)
            self._prob = self._wf * self._wf

            # Compute the local energy estimator and average Energy
            eloc, self._stats = self._get_mc_stats(self._ham)

            # Perform update
            # Derivatives
            for i, state in enumerate(self._states):
                self._der_logs[i] = self._machine.der_log(state)

            # Center the local energy
            eloc -= self._stats['Mean']

            # Center the log derivatives
            mean = self._prob @  self._der_logs
            _MPI.COMM_WORLD.Allreduce(_MPI.IN_PLACE, mean, op=_MPI.SUM)
            self._der_logs -= mean

            # Compute the gradient
            self._grads = _np.conjugate(
                self._der_logs) * (eloc.reshape(-1, 1) * self._prob.reshape(-1, 1))

            grad = _np.sum(self._grads, axis=0)
            _MPI.COMM_WORLD.Allreduce(_MPI.IN_PLACE, grad, op=_MPI.SUM)

            if self._sr:
                dp = _np.empty(self._npar, dtype=_np.complex128)

                self._sr.compute_update(
                    _np.sqrt(self._dim) * self._der_logs * self._wf.reshape(-1, 1), grad, dp)

            else:
                dp = grad

            self._machine.parameters = self._optimizer_step(
                self.step_count, dp, self._machine.parameters
            )

            self.step_count += 1

    def iter(self, n_steps, step=1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_iter (int=None): The total number of steps to perform.
            step_size (int=1): The number of internal steps the simulation
                is advanced every turn.

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            self.advance(step)
            yield self.step_count

    def add_observable(self, obs, name):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.
        """
        self._obs[name] = obs

    def get_observable_stats(self, observables=None, include_energy=True):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.

        Args:
            observables: A dictionary of the form {name: observable} or a list
                of tuples (name, observable) for which statistics should be computed.
                If observables is None or not passed, results for those observables
                added to the driver by add_observables are computed.
            include_energy: Whether to include the energy estimate (which is already
                computed as part of the VMC step) in the result.

        Returns:
            A dictionary of the form {name: stats} mapping the observable names in
            the input to corresponding Stats objects.

            If `include_energy` is true, then the result will further contain the
            energy statistics with key "Energy".
        """
        if not observables:
            observables = self._obs
        r = {"Energy": self._stats} if include_energy else {}

        r.update(
            {name: self._get_mc_stats(obs)[1]
             for name, obs in observables.items()}
        )
        return r

    def reset(self):
        self.step_count = 0

    def _get_mc_stats(self, op):
        loc = _local_values(op, self._machine, self._states)
        mean = self._prob @ loc.reshape(-1, 1)
        _MPI.COMM_WORLD.Allreduce(_MPI.IN_PLACE, mean, op=_MPI.SUM)
        return loc, {"Mean": complex(mean)}

    def __repr__(self):
        return "Vmc(step_count={})".format(
            self.step_count
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian", self._ham),
                ("Machine", self._machine),
                ("Optimizer", self._optimizer_desc),
                ("SR solver", self._sr),
            ]
        ]
        return "\n  ".join([str(self)] + lines)
