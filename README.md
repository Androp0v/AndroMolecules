# AndroMolecules
Package containing useful functions for molecular simulations.

## How to use
```
import andromolecules as am

```

## Updating package

### Editing code in developer mode
Use (may require ```sudo``` in macOS):
```
python3 -m pip install -e .
```

You can also build it with:
```
python3 setup.py build_ext --inplace
```

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