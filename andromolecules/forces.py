# AndroMolecules

import numbers
import numpy
from andromolecules._potentials import _lennard_jones_potential

def lennard_jones_force(
        positions, 
        epsilon, 
        sigma):
    """
    Computes the Lennard-Jones forces of a system of particles.

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
    out : ndarray
        Forces on each of the particles in the system due to Lennard-Jones
        interactions.

    """

    raise NotImplementedError("Not implemented.")
    