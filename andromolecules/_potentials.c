#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "_potentials.h"

/*================== MODULE CONFIG ==================*/

static PyMethodDef _potentials_methods[] = { 
    {   
        "_lennard_jones_potential", 
        _lennard_jones_potential, 
        METH_VARARGS,
        "Lennard-Jones potential.",
    },
    {
        "_lennard_jones_potential_mixed", 
        _lennard_jones_potential_mixed, 
        METH_VARARGS, 
        "Lennard-Jones potential using Lorentz-Berthelot mixing rules."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _potentials_definition = { 
    PyModuleDef_HEAD_INIT,
    "andromolecules._potentials",
    "An internal submodule for common molecular simulation potentials.",
    -1, 
    _potentials_methods
};

PyMODINIT_FUNC PyInit__potentials(void) {
    // Call Py_Initialize to be able to use Python API
    Py_Initialize();

    // Call import_array to be able to use NumPy C API
    import_array();

    // Create the Python module from the definition object
    return PyModule_Create(&_potentials_definition);
};

/*================== FUNCTIONS ==================*/

static PyObject *_lennard_jones_potential(PyObject *self, PyObject *args) {
        
    PyArrayObject *positions;
    double epsilon, sigma, totalEnergy;

    if (!PyArg_ParseTuple(args, "O!dd", 
                          &PyArray_Type, &positions,
                          &epsilon,
                          &sigma)) {
        return NULL;
    }
    
    int positionCount = positions->dimensions[0];

    totalEnergy = 0;

    for (npy_intp i = 0; i < positionCount; i++) {
        for (npy_intp j = 0; j < i; j++) {

            if (i == j) {
                break;
            }

            npy_float64 *positionPointer1 = (double *) PyArray_GetPtr(positions, &i);
            npy_float64 *positionPointer2 = (double *) PyArray_GetPtr(positions, &j);

            // Compute distance
            double distance = fabs(*positionPointer1 - *positionPointer2);

            totalEnergy += 4 * epsilon * ( pow(sigma / distance, 12) 
                - pow(sigma / distance, 6));
        }
    }

    PyObject *result = PyFloat_FromDouble(totalEnergy);
    return result;
}

static PyObject *_lennard_jones_potential_mixed(PyObject *self, PyObject *args) {

    PyArrayObject *positions, *epsilon_array, *sigma_array;
    double totalEnergy, mixed_epsilon, mixed_sigma;

    if (!PyArg_ParseTuple(args, "O!dd", 
                          &PyArray_Type, &positions,
                          &epsilon_array,
                          &sigma_array)) {
        return NULL;
    }

    int positionCount = positions->dimensions[0];

    totalEnergy = 0;

    for (npy_intp i = 0; i < positionCount; i++) {
        for (npy_intp j = 0; j < i; j++) {

            if (i == j) {
                break;
            }

            npy_float64 *positionPointer1 = (double *) PyArray_GetPtr(positions, &i);
            npy_float64 *positionPointer2 = (double *) PyArray_GetPtr(positions, &j);
            npy_float64 *epsilonPointer1 = (double *) PyArray_GetPtr(epsilon_array, &i);
            npy_float64 *epsilonPointer2 = (double *) PyArray_GetPtr(epsilon_array, &j);
            npy_float64 *sigmaPointer1 = (double *) PyArray_GetPtr(sigma_array, &i);
            npy_float64 *sigmaPointer2 = (double *) PyArray_GetPtr(sigma_array, &j);

            // Compute approximate epsilon and sigma using Lorentz-Berthelot
            // mixing rules.
            mixed_epsilon = sqrt(*epsilonPointer1 * *epsilonPointer2);
            mixed_sigma = 0.5 * (*sigmaPointer1 + *sigmaPointer2);

            // Compute distance
            double distance = fabs(*positionPointer1 - *positionPointer2);

            totalEnergy += 4 * mixed_epsilon * ( pow(mixed_sigma / distance, 12) 
                - pow(mixed_sigma / distance, 6));
        }
    }

    PyObject *result = PyFloat_FromDouble(totalEnergy);
    return result;
}










