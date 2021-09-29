#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "_forces.h"

/*================== MODULE CONFIG ==================*/

static PyMethodDef _forces_methods[] = { 
    {   
        "_lennard_jones_force", 
        _lennard_jones_force, 
        METH_VARARGS,
        "Lennard-Jones force.",
    },
    {
        "_lennard_jones_force_mixed", 
        _lennard_jones_force_mixed, 
        METH_VARARGS, 
        "Lennard-Jones force using Lorentz-Berthelot mixing rules."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _forces_definition = { 
    PyModuleDef_HEAD_INIT,
    "andromolecules._forces",
    "An internal submodule for common molecular simulation forces.",
    -1, 
    _forces_methods
};

PyMODINIT_FUNC PyInit__forces(void) {
    // Call Py_Initialize to be able to use Python API
    Py_Initialize();

    // Call import_array to be able to use NumPy C API
    import_array();

    // Create the Python module from the definition object
    return PyModule_Create(&_forces_definition);
};

/*================== FUNCTIONS ==================*/

static PyObject *_lennard_jones_force(PyObject *self, PyObject *args) {
        
    PyArrayObject *positions;
    double epsilon, sigma;

    if (!PyArg_ParseTuple(args, "O!dd", 
                          &PyArray_Type, &positions,
                          &epsilon,
                          &sigma)) {
        return NULL;
    }
    
    int positionCount = positions->dimensions[0];

    // TO-DO

    PyObject *result = PyFloat_FromDouble(0);
    return result;
}

static PyObject *_lennard_jones_force_mixed(PyObject *self, PyObject *args) {

    PyArrayObject *positions, *epsilon_array, *sigma_array;
    double mixed_epsilon, mixed_sigma;

    if (!PyArg_ParseTuple(args, "O!dd", 
                          &PyArray_Type, &positions,
                          &epsilon_array,
                          &sigma_array)) {
        return NULL;
    }

    int positionCount = positions->dimensions[0];

    // TO-DO

    PyObject *result = PyFloat_FromDouble(0);
    return result;
}










