# AndroMolecules

from setuptools import setup, find_packages, Extension

VERSION = '0.1' 
DESCRIPTION = 'Package containing useful functions for molecular simulations'
LONG_DESCRIPTION = 'Package containing useful functions for molecular simulations'

setup(
        name = "andromolecules", 
        version = VERSION,
        author = "Raúl Montón",
        author_email = "<raul.monton@unknown.invalid>",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages(),
        ext_modules=[Extension('andromolecules.odes', ['andromolecules/odes.c'])],
        install_requires = ['numpy>=1.19'], # Older NumPy versions may also work
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)