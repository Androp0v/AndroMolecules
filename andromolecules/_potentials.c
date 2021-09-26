#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "_potentials.h"

/*================== MODULE CONFIG ==================*/
static PyMethodDef _potentials_methods[] = { 
    {   
        "_lennard_jones_potential", _lennard_jones_potential, METH_VARARGS,
        "Lennard-Jones potenital."
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
            npy_float64 *positionPointer = (double *) PyArray_GetPtr(positions, &i);
            double distance = *positionPointer;

            totalEnergy += 4 * epsilon * ( pow(sigma / distance, 12) 
                - pow(sigma / distance, 6));
        }
    }

    PyObject *result = PyFloat_FromDouble(totalEnergy);
    return result;
}