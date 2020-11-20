Lattice Boltzmann with Python
=============================

This repository contains demonstration codes of how to implement and optimize
an MPI parallel, two-dimensional (D2Q9) Lattice Boltzmann solver in Python.

The codes provided here come in three flavors:
- *opt0* is the bare implementation of the underlying mathematical formulation
in Python+numpy.
- *opt1* contains optimizations of this Python code.
- In *opt2*, part of the Python code has been moved to C++. 

A detailed description of this code has been published in the Proceedings of
the 5th bwHPC Symposium that can be found [here](https://10.15496/publikation-29062).

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://travis-ci.org/IMTEK-Simulation/LBWithPython.svg?branch=master)](https://travis-ci.org/IMTEK-Simulation/LBWithPython)

Requirements
------------

To run the codes here you need [numpy](https://www.numpy.org/), [pybind11](https://github.com/pybind/pybind11) and [Eigen](https://eigen.tuxfamily.org/). To run the pure C++ reference implementation you additional need [boost](https://www.boost.org/). 
