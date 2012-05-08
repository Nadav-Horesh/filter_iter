#
# An example of a cython callback function as a filter-kernel
# for gen_filter.py: A random selection of an item from an array
#
# Nadav Horesh 15 Apr 2012
#
#from filter_iter cimport _filter_iter as FI #cimport FloatKernel
cimport _filter_iter ##as FI #cimport FloatKernel
from filter_iter import _filter_iter


cdef extern from "stdlib.h":
    int random()


cdef class Cy_rand_select(_filter_iter.FloatKernel):
    def __init__(self):
        pass
    cdef float evaluate(self, float* buf, ssize_t buflen)  except *:
        cdef int idx
        idx = random() % buflen
        return buf[idx]
