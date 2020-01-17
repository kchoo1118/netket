import numpy as _np
from .abstract_sampler import AbstractSampler

from .._C_netket import sampler as c_sampler
from .._C_netket.utils import random_engine, rand_uniform_real


class PyMetropolisHastings(AbstractSampler):
    """
    ``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
    a local transition kernel to perform moves in the Markov Chain.
    The transition kernel is used to generate
    a proposed state :math:`s^\prime`, starting from the current state :math:`s`.
    The move is accepted with probability

    .. math::
    A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),

    where the probability being sampled is :math:`F(\Psi(s))` (by default :math:`F(x)=|x|^2`)
    and :math:`L(s,s^\prime)` is a correcting factor computed by the transition kernel.
    """

    def __init__(
        self, machine, transition_kernel, n_chains=16, sweep_size=None, batch_size=None
    ):
        """
        Constructs a new ``MetropolisHastings`` sampler given a machine and
        a transition kernel.

        Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                          The probability distribution being sampled
                          from is :math:`F(\Psi(s))`, where the function
                          $$F(X)$$, is arbitrary, by default :math:`F(X)=|X|^2`.
            transition_kernel: A function to generate a transition.
                          This should take as an input the current state (in batches)
                          and return a modified state (also in batches).
                          This function must also return an array containing the
                          `log_prob_corrections` :math:`L(s,s^\prime)`.
            n_chains: The number of Markov Chain to be run in parallel on a single process.
            sweep_size: The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number Markov chains (n_chains).

        """

        self.machine = machine
        self.n_chains = n_chains

        self.sweep_size = sweep_size

        self._kernel = transition_kernel

        self.machine_func = lambda x, out=None: _np.square(
            _np.absolute(x), out)

        super().__init__(machine, n_chains)

    @property
    def n_chains(self):
        return self._n_chains

    @n_chains.setter
    def n_chains(self, n_chains):
        if n_chains < 0:
            raise ValueError("Expected a positive integer for n_chains ")

        self._n_chains = n_chains

        self._state = _np.zeros((n_chains, self._n_visible))
        self._state1 = _np.copy(self._state)

        self._log_values = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_prob_corr = _np.zeros(n_chains)

    @property
    def machine_func(self):
        return self._machine_func

    @machine_func.setter
    def machine_func(self, machine_fun):
        self._machine_func = machine_fun

    @property
    def sweep_size(self):
        return self._sweep_size

    @sweep_size.setter
    def sweep_size(self, sweep_size):
        self._sweep_size = sweep_size if sweep_size != None else self._n_visible
        if self._sweep_size < 0:
            raise ValueError("Expected a positive integer for sweep_size ")
        self._rand_for_acceptance = _np.zeros(
            self._sweep_size * self.n_chains, dtype=float
        )

    @property
    def machine(self):
        return self._machine

    @machine.setter
    def machine(self, machine):
        self._machine = machine
        self._n_visible = machine.hilbert.size
        self._hilbert = machine.hilbert

    def reset(self, init_random=False):
        if init_random:
            for state in self._state:
                self._hilbert.random_vals(state, random_engine())
        self.machine.log_val(self._state, out=self._log_values)

    def _log_val_batched(self, v, out=None):
        return self.machine.log_val(v, out)

    def __next__(self):

        rand_uniform_real(self._rand_for_acceptance)

        for sweep in range(self.sweep_size):

            # Propose a new state using the transition kernel
            self._kernel(self._state, self._state1, self._log_prob_corr)

            self._log_val_batched(self._state1, out=self._log_values_1)

            # Acceptance probability
            self._prob = self.machine_func(
                _np.exp(self._log_values_1 -
                        self._log_values + self._log_prob_corr)
            )

            # Acceptance test
            accept = self._prob > self._rand_for_acceptance[sweep]

            # Update of the state
            self._log_values = _np.where(
                accept, self._log_values_1, self._log_values)
            self._state = _np.where(
                accept.reshape(-1, 1), self._state1, self._state)
        return self._state, self._log_values


class MetropolisLocal(AbstractSampler):
    """
    Sampler acting on one local degree of freedom.

    This sampler acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \\neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen
    with uniform probability.
    2. Among all the possible (:math:`m`) values that :math:`s_i` can take,
    one of them is chosen with uniform probability.

    For example, in the case of spin :math:`1/2` particles, :math:`m=2`
    and the possible local values are :math:`s_i = -1,+1`.
    In this case then :class:`MetropolisLocal` is equivalent to flipping a random spin.

    In the case of bosons, with occupation numbers
    :math:`s_i = 0, 1, \dots n_{\mathrm{max}}`, :class:`MetropolisLocal`
    would pick a random local occupation number uniformly between :math:`0`
    and :math:`n_{\mathrm{max}}`.
    """

    def __init__(
        self, machine, n_chains=16, sweep_size=None, batch_size=None, backend=None
    ):
        """

         Constructs a new :class:`MetropolisLocal` sampler given a machine.

         Args:
            machine: A machine :math:`\Psi(s)` used for the sampling.
                     The probability distribution being sampled
                     from is :math:`F(\Psi(s))`, where the function
                     :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.

            n_chains:   The number of Markov Chain to be run in parallel on a single process.
            sweep_size: The number of exchanges that compose a single sweep.
                        If None, sweep_size is equal to the number of degrees of freedom (n_visible).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                        If None, batch_size is equal to the number Markov chains (n_chains).


         Examples:
             Sampling from a RBM machine in a 1D lattice of spin 1/2

             >>> import netket as nk
             >>>
             >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
             >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
             >>>
             >>> # RBM Spin Machine
             >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
             >>>
             >>> # Construct a MetropolisLocal Sampler
             >>> sa = nk.sampler.MetropolisLocal(machine=ma)
             >>> print(sa.machine.hilbert.size)
             100
        """
        if "_C_netket.machine" in str(type(machine)) and backend != "py":
            self.sampler = c_sampler.MetropolisLocal(
                machine=machine,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.LocalKernel(machine.hilbert),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisLocalPt(AbstractSampler):
    """
    """

    def __init__(self, machine, n_replicas=16, sweep_size=None, batch_size=None):
        """

        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisLocalPt(
                machine=machine, n_replicas=n_replicas, sweep_size=sweep_size
            )
        else:
            raise ValueError(
                """Parallel Tempering samplers are not yet implemented
                for pure python machines"""
            )
        super().__init__(machine, 1)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisExchange(AbstractSampler):
    """
    """

    def __init__(self, machine, d_max=1, n_chains=16, sweep_size=None, batch_size=None):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchange(
                machine=machine,
                n_chains=n_chains,
                d_max=d_max,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.ExchangeKernel(machine.hilbert, d_max),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisExchangeChemistry(AbstractSampler):
    """
    """

    def __init__(self, machine, npar, particle_hole, njumps=1, n_chains=16, sweep_size=None, batch_size=None):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchangeChemistry(
                machine=machine,
                npar=npar,
                particle_hole=particle_hole,
                njumps=njumps,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.ExchangeChemistryKernel(
                    machine.hilbert, npar, particle_hole, njumps),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisExchangePt(AbstractSampler):
    """
    """

    def __init__(
        self, machine, d_max=1, n_replicas=16, sweep_size=None, batch_size=None
    ):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisExchangePt(
                machine=machine,
                n_replicas=n_replicas,
                d_max=d_max,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            raise ValueError(
                """Parallel Tempering samplers are not yet implemented
                for pure python machines"""
            )
        super().__init__(machine, 1)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisHamiltonian(AbstractSampler):
    """
    """

    def __init__(
        self, machine, hamiltonian, n_chains=16, sweep_size=None, batch_size=None
    ):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisHamiltonian(
                machine=machine,
                hamiltonian=hamiltonian,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.HamiltonianKernel(hamiltonian),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class MetropolisHamiltonianPt(AbstractSampler):
    """
    """

    def __init__(self, machine, hamiltonian, n_replicas=16, sweep_size=None):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.MetropolisHamiltonianPt(
                machine=machine,
                hamiltonian=hamiltonian,
                n_replicas=n_replicas,
                sweep_size=sweep_size,
            )
        else:
            raise ValueError(
                """Parallel Tempering samplers are not yet implemented
                for pure python machines"""
            )
        super().__init__(machine, 1)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class CustomSampler(AbstractSampler):
    """
    """

    def __init__(
        self,
        machine,
        move_operators,
        move_weights=None,
        n_chains=16,
        sweep_size=None,
        batch_size=None,
    ):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.CustomSampler(
                machine=machine,
                move_operators=move_operators,
                move_weights=move_weights,
                n_chains=n_chains,
                sweep_size=sweep_size,
                batch_size=batch_size,
            )
        else:
            self.sampler = PyMetropolisHastings(
                machine,
                c_sampler.CustomLocalKernel(move_operators, move_weights),
                n_chains,
                sweep_size,
                batch_size,
            )
        super().__init__(machine, n_chains)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func


class CustomSamplerPt(AbstractSampler):
    """
    """

    def __init__(
        self, machine, move_operators, move_weights=None, n_replicas=16, sweep_size=None
    ):
        """
        """
        if "_C_netket.machine" in str(type(machine)):
            self.sampler = c_sampler.CustomSamplerPt(
                machine=machine,
                move_operators=move_operators,
                move_weights=move_weights,
                n_replicas=n_replicas,
                sweep_size=sweep_size,
            )
        else:
            raise ValueError(
                """Parallel Tempering samplers are not yet implemented
                            for pure python machines"""
            )
        super().__init__(machine, 1)

    def reset(self, init_random=False):
        self.sampler.reset(init_random)

    def __next__(self):
        return self.sampler.__next__()

    @property
    def machine_func(self):
        return self.sampler.machine_func

    @machine_func.setter
    def machine_func(self, func):
        self.sampler.machine_func = func
