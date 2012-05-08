"""
A platform for faster image filters
Nadav Horesh
"""

from __future__ import print_function
import numpy as np
import cPickle
cimport numpy as np

# For the Cython functions interface

cdef class DoubleKernel:
    cdef double evaluate(self, double* buf, ssize_t buflen) except *:
        raise NotImplementedError()

cdef class FloatKernel:
    cdef float evaluate(self, float* buf, ssize_t buflen) except *:
        raise NotImplementedError()


#ctypedef from  "filter_header.h":
#    float* float32_fcn(float, int)
ctypedef float float32_fcn(float*, int)
ctypedef double float64_fcn(double*, int)
ctypedef unsigned char byte_fcn(unsigned char*, int)
ctypedef unsigned int uint


cdef extern from "strings.h":
   void *memcpy(char *dest, char *src, size_t n)


cdef copy_slice_to_vec(np.ndarray src,
                       uint first_row, uint first_col,
                       uint width, uint height,
                       uint itemsize, char* to):
    cdef uint llen =  src.shape[1] * itemsize
    cdef uint offset
    cdef char* pad_dat = <char *> src.data
    cdef uint copy_len = width * itemsize
    for i in range(height):
        offset = (first_row + i)*llen + first_col*itemsize
        memcpy(&to[i*copy_len], &pad_dat[offset], copy_len)



class gen_filter_matrix(object):
    """
    Generate a matrix for neighbourhood iteration and provide iterators
    """
    def __init__(self, data, kernel_shape, padding=None, mode='constant', cval=0.0):
        """
        Generate a 2D array for iterations

        Parameters
        ----------
        data : 2D array
            Data to filter
        kernel_shape : A length 2 sequence of integers
            Shape of the filter kernel
        mode : str
           Padding mode see ndimage.generic_filter for details
        padding : None or int
            Array padding for the neighborhood iterations.
        """
        if data.ndim != 2:
            raise ValueError('Input array must be 2 dimentional')
        shape = data.shape
        self.shape = shape
        self.ker_shape = kernel_shape
        if padding is None:
            padding = kernel_shape[0]//2, kernel_shape[1]//2
            pad_shape = [shape[i]+kernel_shape[i]-1 for i in (0,1)]
            self.origin = padding
        else:
            pad_shape = [shape[i]+2*padding for i in (0,1)]
            self.origin = padding, padding
        self.mode = mode
        if padding == 0:
            self.padded = data
        else:
            self.padded = np.zeros(pad_shape, dtype=data.dtype)
            if mode == 'constant':
                self.padded[:] = cval
            self.padded[self.origin[0]:self.origin[0]+shape[0],
                        self.origin[1]:self.origin[1]+shape[1]] = data
            # Fill according to the padding mode:
            origin = self.origin
            if mode == 'mirror':
                for i in range(self.origin[0]):
                    self.padded[self.origin[0]-1-i] = self.padded[self.origin[0]+1+i]
                    self.padded[-self.origin[0]+i] = self.padded[-self.origin[0]-2-i]
                for i in range(self.origin[1]):
                    self.padded[:,self.origin[1]-1-i] = self.padded[:,self.origin[1]+1+i]
                    self.padded[:,-self.origin[1]+i] = self.padded[:,-self.origin[1]-2-i]

            elif mode == 'nearest':
                self.padded[:origin[0]] = self.padded[origin[0]]
                self.padded[-origin[0]:] = self.padded[-origin[0]-1]
                self.padded[:,:origin[1]] = self.padded[:,origin[1]][:,None]
                self.padded[:,-origin[1]:] = self.padded[:,-origin[1]-1][:,None]

            elif mode == 'reflect':
                for i in range(self.origin[0]):
                    self.padded[self.origin[0]-1-i] = self.padded[self.origin[0]+i]
                    self.padded[-self.origin[0]+i] = self.padded[-self.origin[0]-1-i]
                for i in range(self.origin[1]):
                    self.padded[:,self.origin[1]-1-i] = self.padded[:,self.origin[1]+i]
                    self.padded[:,-self.origin[1]+i] = self.padded[:,-self.origin[1]-1-i]

            elif mode == 'wrap':
                for i in range(self.origin[0]):
                    self.padded[-self.origin[0]+i] = self.padded[self.origin[0]+1+i]
                    self.padded[self.origin[0]-1-i] = self.padded[-self.origin[0]-1-i]
                for i in range(self.origin[1]):
                    self.padded[:,-self.origin[1]+i] = self.padded[:,self.origin[1]+1+i]
                    self.padded[:,self.origin[1]-1-i] = self.padded[:,-self.origin[1]-2-i]

            elif mode == 'constant':
                pass
            else:
                raise ValueError('Invalid mode string: "%s"' % mode)

    def py_iterator(self):
        """
        Iterator over the image.
        Return a n array which refere to the original data

        parameters
        ==========
        size: int
          The environment size (size X size)
        """
        pad = self.padded
        ks0, ks1 = self.ker_shape
        return (pad[i:i+ks0, j:j+ks1]
                    for i in range(self.shape[0]) for j in range(self.shape[1]))

    def py_iterator_copy(self):
        """
        Iterator over the image.
        Return an array which is a copy from the padded image

        parameters
        ==========
        size: int
          The environment size (size X size)
        """
        pad = self.padded
        ks0, ks1 = self.ker_shape
        return (pad[i:i+ks0, j:j+ks1].copy()
                    for i in range(self.shape[0]) for j in range(self.shape[1]))

    def filter_with_py_callback(self, pycall_back, output=None, dtype=None):
        """
        Iterate a given python function over the padded data.

        parameters:
        ===========
        pycall_back : A python callable object
          A python callable oblect which is provided with a 2D slice
          of the padded array.
        output: A 2D array (optional)
          output array. If not provided, a new array is generated.
        dtype : A numpy dtype (optional)
          The dtype of the output array, If None, it set to the dtype of
          the original array. dtype is ingnoed if outout arrray is provided.
        """
        if dtype is None:
            dtype = self.padded.dtype
        if output is None:
            output = np.empty(self.shape, dtype=dtype)
        pad = self.padded
        ks0, ks1 = self.ker_shape

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                output[i,j] = pycall_back(pad[i:i+ks0, j:j+ks1])
        return output

##    C level methods

    def filter_with_C_callback_float(self, ssize_t C_callback,
##    def filter_with_C_callback_float(self, float32_fcn* C_callback,
                                     np.ndarray[ndim=2, dtype=float] output not None):
        """
        Apply a C function call-back on the padded array.

        parameters
        ==========

        C_callback : unsigned long int
          The address of a C function. A major wrinkle here that the
           address is provided as an unsigned long and not as a pointer.
           (does anyone knows how to provide a pointer from python as is
           an not to camouflage it as an integer?

        the call-back receives two parameters: the first is a pointer
        to a buffer and the second (integer type) is a length of the buffer.
        """
        cdef unsigned int i, j, ks0, ks1
        cdef float32_fcn* call_back = <float32_fcn*> C_callback
        cdef int buf_size = self.ker_shape[0]*self.ker_shape[1]
        cdef np.ndarray[ndim=1, dtype=float] neigh = np.empty(buf_size, dtype=np.float32)
        #cdef np.ndarray[ndim=2, dtype=float] output = np.empty(self.shape, dtype=np.float32)
        cdef float* cp = <float*> neigh.data
        ks0, ks1 = self.ker_shape

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                #neigh[:] = self.padded[i:i+ks0, j:j+ks1].ravel()
                copy_slice_to_vec(self.padded, i, j, ks0, ks1, 4, neigh.data)
                output[i,j] = call_back(cp, buf_size)
        return output


    def filter_with_C_callback_double(self, unsigned long int C_callback,
                                     np.ndarray[ndim=2, dtype=double] output not None):
        """
        Apply a C function call-back on the padded array.

        parameters
        ==========

        C_callback : unsigned long int
          The address of a C function. A major wrinkle here that the
           address is provided as an unsigned long and not as a pointer.
           (does anyone knows how to provide a pointer from python as is
           an not to camouflage it as an integer?

        the call-back receives two parameters: the first is a pointer
        to a buffer and the second (integer type) is a length of the buffer.
        """
        cdef unsigned int i, j, ks0, ks1
        cdef float64_fcn* call_back = <float64_fcn*> C_callback
        cdef int buf_size = self.ker_shape[0]*self.ker_shape[1]
        cdef np.ndarray[ndim=1, dtype=double] neigh = np.empty(buf_size, dtype=np.float64)
        #cdef np.ndarray[ndim=2, dtype=float] output = np.empty(self.shape, dtype=np.float32)
        cdef double* cp = <double*> neigh.data
        ks0, ks1 = self.ker_shape

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                neigh[:] = self.padded[i:i+ks0, j:j+ks1].ravel()
                output[i,j] = call_back(cp, buf_size)
        return output


###########################################################################################
#
#  Cython callback

    def filter_with_Cy_callback_float(self, FloatKernel Cy_callback,
                                     np.ndarray[ndim=2, dtype=float] output = None):
        """
        Apply a C function call-back on the padded array.

        parameters
        ==========

        Cy_call_back : A cython extension type (prototyped in filter_iter.pxd
          (FloatKernel class). the Cy_call_back callback class must contain
          "evaluate" method according the filter_iter.pxd definition:
              cdef float evaluate(self, float* buf, size_t buflen):

        """
        cdef unsigned int i, j, ks0, ks1
        cdef int buf_size = self.ker_shape[0]*self.ker_shape[1]
        cdef np.ndarray[ndim=1, dtype=float] neigh = np.empty(buf_size, dtype=np.float32)
        #cdef np.ndarray[ndim=2, dtype=float] output = np.empty(self.shape, dtype=np.float32)
        cdef float* cp = <float*> neigh.data
        ks0, ks1 = self.ker_shape

        if output is None:
            output = np.empty(self.shape, dtype=np.float32)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                copy_slice_to_vec(self.padded, i, j, ks0, ks1, 4, neigh.data)
                output[i,j] = Cy_callback.evaluate(cp, buf_size)
        return output
