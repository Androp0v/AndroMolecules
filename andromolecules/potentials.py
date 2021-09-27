# AndroMolecules

import numpy
from andromolecules._potentials import _lennard_jones_potential

def lennard_jones_potential(positions, epsilon, sigma):
    """
    Returns the Lennard-Jones potential of a system of particles.
    """
    if not isinstance(positions, numpy.ndarray):
        raise TypeError("'Positions' argument is not a NumPy array.")

    if positions.dtype != numpy.float64:
        raise NotImplementedError("Only float64 NumPy arrays are supported.")
    
    if not positions.flags.contiguous:
        raise NotImplementedError("Only contiguous NumPy arrays are supported.")

    return _lennard_jones_potential(positions, epsilon, sigma)
    