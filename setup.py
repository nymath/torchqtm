from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy
import glob

extensions = glob.glob("quant/_C/*.pyx")

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    name="torchqtm",
    version="0.0.1",
    author="ny",
    author_email="nymath@163.com",
    description="None",
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nymath/torchquantum/tree/main",
    packages=['quant'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    python_requires='>=3.6',
)


## setup cython
# python setup.py build_ext --inplace
## setup package
# pip install twine
# python setup.py sdist bdist_wheel
# twine upload dist/*
# twine upload --skip-existing dist/*