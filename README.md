Lattice Boltzmann with Python
=============================

This repository contains demonstration codes of how to implement and optimize
an MPI parallel, two-dimensional (D2Q9) Lattice Boltzmann solver in Python.

The codes provided here come in three flavors:
- *opt0* is the bare implementation of the underlying mathematical formulation
in Python+numpy.
- *opt1* contains optimizations of this Python code.
- In *opt2*, part of the Python code has been moved to C++. 

Requirements
------------

To run the codes here you need [numpy](https://www.numpy.org/), [pybind11](https://github.com/pybind/pybind11) and [Eigen](https://eigen.tuxfamily.org/). To run the pure C++ reference implementation you additional need [boost](https://www.boost.org/). 