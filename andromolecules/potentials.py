# AndroMolecules

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
    epsilon : float64, dict
        Intensity of the Lennard-Jones potential (depth of the potential well), 
        in eV units. If the system is composed of particles of different types, 
        epsilon is a dict where keys are the particle types and values are the 
        epsilon values for that particle type.
    sigma : float64, dict
        Length at which the Lennard-Jones potential between two atoms becomes 0,
        in Å units. If the system is composed of particles of different types, 
        sigma is a dict where keys are the particle types and values are the 
        sigma values for that particle type.
    particle_types : NoneType, list, ndarray
        If None, particles in the system are assumed to be identical and epsilon
        and sigma are identical for all particles. If particle_types is a list 
        or a ndarray it must be the same length as positions in a way that the 
        ith component of the particle_types array describes the type of the ith
        particle. The potential between two particles A and B is computed using
        the Lorentz-Berthelot mixing rules.

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

    if particle_types != None:
        if not isInstance(epsilon, dict):
            raise TypeError("Epsilon parameter must be a dict when particle_types is not None.")
        if not isInstance(epsilon, dict):
            raise TypeError("Sigma parameter must be a dict when particle_types is not None.")

        epsilon_array = numpy.empty(len(particle_types))
        sigma_array = numpy.empty(len(particle_types))

        for i, particle_type in enumerate(particle_types):
            epsilon_array[i] = epsilon[particle_type]
            sigma_array[i] = sigma[particle_type]

        raise _lennard_jones_potential_mixed(positions, epsilon_array, sigma_array)
    else:
        if not epsilon.isNumeric():
            raise TypeError("Epsilon parameter must be a numerical type")
        if not sigma.isNumeric():
            raise TypeError("Sigma parameter must be a numerical type")

        return _lennard_jones_potential(positions, epsilon, sigma)
    