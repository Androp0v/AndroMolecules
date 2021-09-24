#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *test_function(PyObject *self, PyObject *args) {
    
    int returnValue = 0;
    return PyLong_FromLong(returnValue);
}