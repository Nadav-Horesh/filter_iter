cdef class FloatKernel:
    cdef float evaluate(self, float* buf, ssize_t buflen) except *


cdef class DoubleKernel:
    cdef double evaluate(self, double* buf, ssize_t buflen) except *
