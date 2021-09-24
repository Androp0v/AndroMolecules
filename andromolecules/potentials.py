# AndroMolecules

def lennard_jones_potential(positions, epsilon, sigma):
    """
    Returns the Lennard-Jones potential of a system of particles.
    """
    return 48 * epsilon * np.power(sigma, 12) / np.power(r, 13) - 24 \
    * epsilon * np.power(sigma, 6) / np.power(r, 7)
