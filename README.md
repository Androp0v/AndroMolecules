# AndroMolecules
Package containing useful functions for molecular simulations, performing as fast as I could make them.

## How to use
Example: compute Lennard-Jones potential of a system of particles in 1D.
```
import andromolecules as am
import numpy as np

# Create 10 particles randomly over a line 100 armstrongs long
particle_positions = np.random.random(10) * 100

# Define the parameters of the Lennard-Jones potential
epsilon = 1
sigma = 1

# Compute the Lennard-Jones potential of the system
lennard_jones_energy = am._potentials._lennard_jones_potential(particle_positions, epsilon, sigma)
```

## Contributing

### Editing code in developer mode
Use (may require `sudo` in macOS):
```
python3 -m pip install -e .
```

You can also build it with:
```
python3 setup.py build_ext --inplace
```

### Adding new C extension files
Edit `setup.py` to add a new `Extension` under `ext_modules`:
```
Extension('andromolecules.submodule.file', ['andromolecules/submodule/file.c'])
```
And then create the referred C files and headers in the specified location.

### Create new source code and wheel
Run:
```
python3 setup.py sdist bdist_wheel
```

### Upload new build to PyPi
Run:
```
twine upload dist/*
```

### Style guide
For Python style guides, refer to [PEP8](https://www.python.org/dev/peps/pep-0008/), with one exception:

> Don't use spaces around the = sign when used to indicate a keyword argument, or when used to indicate a default value for an unannotated function parameter.

Using spaces aroung the = sign in function arguments actually improves readability IMHO.

### Useful resources online
[Numpy C Code explanations](https://numpy.org/doc/stable/reference/internals.code-explanations.html)