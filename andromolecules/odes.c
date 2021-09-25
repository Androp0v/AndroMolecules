#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "odes.h"

/*================== MODULE CONFIG ==================*/
static PyMethodDef odes_methods[] = { 
    {   
        "runge_kutta_4", runge_kutta_4, METH_VARARGS,
        "Runge-Kutta 4th order C extension."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef odes_definition = { 
    PyModuleDef_HEAD_INIT,
    "odes",
    "A Python (sub?)module for solving Ordinary Differential Equations.",
    -1, 
    odes_methods
};

PyMODINIT_FUNC PyInit_odes(void) {
    // Call Py_Initialize to be able to use Python API
    Py_Initialize();

    // Call import_array to be able to use NumPy C API
    import_array();

    // Create the Python module from the definition object
    return PyModule_Create(&odes_definition);
};

/*================== FUNCTIONS ==================*/

static PyObject *runge_kutta_4(PyObject *self, PyObject *args) {
        
    PyArrayObject *positions, *forces;

    if (!PyArg_ParseTuple(args, "O!O!", 
                          &PyArray_Type, &positions, 
                          &PyArray_Type, &forces)) {
        return NULL;
    }
    
    int positionCount = positions->dimensions[0];
    int forceCount = forces->dimensions[0];

    if (positionCount != forceCount) {
        // Abort if the number of forces is different to the number of positions
        return NULL;
    }

    /// Main loop

    #pragma omp parallel
    for (npy_intp i = 0; i < positionCount; i++) {
        npy_float64 *positionPointer = (double *) PyArray_GetPtr(positions, &i);
        *positionPointer += 1;
        npy_float64 *forcePointer = (double *) PyArray_GetPtr(forces, &i);
        *forcePointer += 1;
    }

    Py_RETURN_NONE;
}