from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob

extensions = glob.glob("quant/_C/*.pyx")

setup(
    ext_modules=cythonize(extensions),
    compiler_directives={'language_level': "3"},
    include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace


