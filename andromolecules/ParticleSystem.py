# AndroMolecules

import numpy

class ParticleSystem():
    """
    This class wraps the data of a system of particles, so properties like 
    distances between particles are only computed once per iteration.
    """
    def __init__(self, positions):
        self.positions = positions

    def add_force(force, **kwargs):
        """
        Adds a force field to the particle system.
        Parameters
        ----------
        force : string, function
            A string with one of the implemented potentials or a function object
            to be called with the arguments specified in **kwargs.

        Other parameters
        ----------
        **kwargs : any
            Pass any additional parameters required for the force function.

        """

        if isinstance(force, str):

            # Check for named forces

            if force == 'lennard-jones':
                raise NotImplementedError("Not implemented")
            elif force == 'coulomb':
                raise NotImplementedError("Not implemented")
            else:
                raise ValueError("Unknown force named in argument force")

        elif callable(force):

            # Add user-defined functions

            self.force_list.append(force)

        else:

            raise AttributeError("Argument force is neither a string nor a function")

    def simulate_system(time, timestep):
        """
        """
        raise NotImplementedError("Not implemented")

