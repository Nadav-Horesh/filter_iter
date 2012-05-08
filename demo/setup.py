# Compile examples in-place

from __future__ import print_function

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import filter_iter
import sys

sys.argv  = [sys.argv[0], 'build_ext',  '-i']
pxd_path = filter_iter.__path__

print('\n\n Compile examples in place.\n\n')

extensions = [
    Extension("example_Cy_filters", ["example_Cy_filters.pyx"],
              include_dirs=pxd_path),
    Extension("example_C_filters", ["example_C_filters.c"]),
    ]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=extensions)
