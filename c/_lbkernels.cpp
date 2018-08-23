/*
Copyright 2017-2018 Lars Pastewka, Andreas Greiner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------------
| LATTICE BOLTZMANN KERNELS |
-----------------------------

C++ kernel containing an implementation of the collision operation. This kernels
require pybind11 and Eigen.
*/

#include "lbkernels.h"
#include "d2q9.h"

PYBIND11_MODULE(_lbkernels, m) {
    m.doc() = "Lattice Boltzmann kernels";

    m.def("equilibrium", &d2q9_equilibrium1<float>,
          "Return the equilibrium distribution function.");
    m.def("equilibrium", &d2q9_equilibriumn<float>,
          "Return the equilibrium distribution function for an array of values.");
    m.def("collide", &d2q9_colliden<float>,
          "Carry out collision operation for an array of values.");

    m.def("equilibrium", &d2q9_equilibrium1<double>,
    	  "Return the equilibrium distribution function.");
    m.def("equilibrium", &d2q9_equilibriumn<double>,
    	  "Return the equilibrium distribution function for an array of values.");
    m.def("collide", &d2q9_colliden<double>,
    	  "Carry out collision operation for an array of values.");
}
