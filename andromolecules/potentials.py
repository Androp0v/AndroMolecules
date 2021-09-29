# AndroMolecules

import numbers
import numpy
from andromolecules._potentials import _lennard_jones_potential

def lennard_jones_potential(
        positions, 
        epsilon, 
        sigma,
        particle_types = None):
    """
    Computes the Lennard-Jones potential of a system of particles.

    Parameters
    ----------
    positions : ndarray
        The array with the positions of all the particles in the system.
    epsilon : float64, ndarray
        Intensity of the Lennard-Jones potential (depth of the potential well), 
        in eV units. If an array is passed, it's interpreted as the epsilon
        value for each particle in the positions array, and interactions are 
        computed using the Lorentz-Berthelot mixing rules.
    sigma : float64, ndarray
        Length at which the Lennard-Jones potential between two atoms becomes 0,
        in Å units. If an array is passed, it's interpreted as the sigma value 
        for each particle in the positions array, and interactions are computed 
        using the Lorentz-Berthelot mixing rules.
    Returns
    -------
    out : float64
        Total potential energy of all Lennard-Jones interactions in the system, 
        in eV units.

    """

    if not isinstance(positions, numpy.ndarray):
        raise TypeError("'Positions' argument is not a NumPy array.")

    if not positions.dtype == numpy.float64:
        raise NotImplementedError("Only float64 NumPy arrays are supported.")
    
    if not positions.flags.contiguous:
        raise NotImplementedError("Only contiguous NumPy arrays are supported.")

    if isinstance(epsilon, numpy.ndarray):
        single_epsilon = False
    elif isinstance(epsilon, numbers.Number):
        single_epsilon = True
    else:
        raise TypeError("'epsilon' argument is not a number nor a NumPy array.")

    if isinstance(sigma, numpy.ndarray):
        single_sigma = False
    elif isinstance(sigma, numbers.Number):
        single_sigma = True
    else:
        raise TypeError("'sigma' argument is not a number nor a NumPy array.")

    if not (single_epsilon and single_sigma):

        # Invoke the C function with epsilon, sigma array arguments

        array_length = len(positions)

        if not single_epsilon:
            epsilon_array = epsilon
        else:
            # Fill the array with the single-valued epsilon
            epsilon_array = numpy.full(array_length)

        if not single_sigma:
            sigma_array = sigma
        else:
            # Fill the array with the single-valued sigma
            sigma_array = numpy.full(array_length)

        return _lennard_jones_potential_mixed(positions, epsilon_array, sigma_array)

    else:

        # Invoke the C function with epsilon, sigma float64 arguments

        if not isinstance(epsilon, numbers.Number):
            raise TypeError("Epsilon parameter must be a numerical type")
        if not isinstance(sigma, numbers.Number):
            raise TypeError("Sigma parameter must be a numerical type")

        return _lennard_jones_potential(positions, epsilon, sigma)
    