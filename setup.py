from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob

extensions = glob.glob("_C/*.pyx")

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace
