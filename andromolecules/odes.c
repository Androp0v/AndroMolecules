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
    "andromolecules.odes",
    "A submodule for solving Ordinary Differential Equations.",
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
        
    PyArrayObject *positions, *velocities, *forces;

    if (!PyArg_ParseTuple(args, "O!O!O!", 
                          &PyArray_Type, &positions,
                          &PyArray_Type, &velocities,
                          &PyArray_Type, &forces)) {
        return NULL;
    }
    
    int positionCount = positions->dimensions[0];
    int velocitiesCount = velocities->dimensions[0];
    int forceCount = forces->dimensions[0];

    if ((positionCount != velocitiesCount) || (positionCount != forceCount)) {
        // Abort if the particle properties arrays don't have the same length
        return NULL;
    }

    /*** Main loop ***/

    // Define the Runge-Kutta coeficcients
    npy_float64 k1x, k2x, k3x, k4x, k1v, k2v, k3v, k4v;
    npy_float64 deltat = 0.01;
    npy_float64 mass = 1.0;

    // Loop over all particles to get 1st RK coefficient
    for (npy_intp i = 0; i < positionCount; i++) {
        npy_float64 *positionPointer = (double *) PyArray_GetPtr(positions, &i);
        npy_float64 *velocityPointer = (double *) PyArray_GetPtr(velocities, &i);
        npy_float64 *forcePointer = (double *) PyArray_GetPtr(forces, &i);

        k1x = *velocityPointer * deltat;
        k1v = *forcePointer / mass * deltat;
    }

    // Loop over all particles to get 2nd RK coefficient
    for (npy_intp i = 0; i < positionCount; i++) {
        npy_float64 *positionPointer = (double *) PyArray_GetPtr(positions, &i);
        npy_float64 *velocityPointer = (double *) PyArray_GetPtr(velocities, &i);
        npy_float64 *forcePointer = (double *) PyArray_GetPtr(forces, &i);

        k1x = *velocityPointer * deltat;
        k1v = *forcePointer / mass * deltat;
        k2x = (*velocityPointer + 0.5 * k1v) * deltat;
        k2v = *forcePointer / mass * deltat;
        //*positionPointer += 1;
        //*forcePointer += 1;
    }

    Py_RETURN_NONE;
}