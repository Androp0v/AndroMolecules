# AndroMolecules
Package containing useful functions for molecular simulations, performing as fast as I could make them.

## How to use
```
import andromolecules as am

# TO-DO
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

### Adding new C files
Edit `setup.py` to add a new `Extension` under `ext_modules`:
```
Extension('andromolecules.submodule.file', ['andromolecules/submodule/file.c'])
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

### Useful resources online
[Numpy C Code explanations](https://numpy.org/doc/stable/reference/internals.code-explanations.html)