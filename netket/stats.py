from ._C_netket.stats import *
from ._C_netket.stats import _subtract_mean
from ._C_netket.stats import _subtract_weighted_mean
from ._C_netket.stats import _l1_norm
from ._C_netket.stats import _l2_norm


def subtract_mean(x):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.
    """
    return _subtract_mean(x.reshape(-1, x.shape[-1]))


def subtract_weighted_mean(x, w):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.
    """
    return _subtract_weighted_mean(x.reshape(-1, x.shape[-1]), w.reshape(1, -1))
