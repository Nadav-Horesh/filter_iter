#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

#import gen_filter
from filter_iter import _filter_iter as fiter, gen_filter
import example_Cy_filters as cy_example

def py_rand_select(buffer):
    buf = buffer.ravel()
    return buf[np.random.randint(len(buf))]


x = np.linspace(-3,3,51).astype(np.float32)
z = np.exp(-(x**2+x[:,None]**2))
z[::2, ::2] = 0.0

C_filtered = gen_filter.apply_C_filter(z, './example_C_filters.so',
                                       'rand_select', (3,3))
prepare = fiter.gen_filter_matrix(z, (3,3))

Py_filtered = prepare.filter_with_py_callback(py_rand_select)

Cy_filtered =  prepare.filter_with_Cy_callback_float(cy_example.Cy_rand_select())

fig, (ax1, ax2, ax3,ax4) = plt.subplots(ncols=4, figsize=(15, 5))
ax1.imshow(z, interpolation='nearest')
ax1.set_title('Original data')

ax2.imshow(C_filtered, interpolation='nearest')
ax2.set_title('C filtered data')

ax3.imshow(Py_filtered, interpolation='nearest')
ax3.set_title('Python filtered data')

ax4.imshow(Cy_filtered, interpolation='nearest')
ax4.set_title('Cython filtered data')

plt.show()
