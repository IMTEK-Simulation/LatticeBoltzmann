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

--------
| OPT2 |
--------

C++ kernel containing an implementation of the collision operation.

This kernel requires pybind11 and Eigen. Compile with
c++ -O3 -Wall -shared -std=c++11 -fPIC -I`python3 -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"` -I/usr/local/Cellar/pybind11/2.2.3/include -I/usr/local/Cellar/eigen/3.3.5/include/eigen3 `pkg-config python3 --libs` shear_wave_opt2_cext.cpp -o shear_wave_opt2_cext`python3-config --extension-suffix`
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <iostream>

constexpr double w_0 = 4./9;
constexpr double w_1234 = 1./9;
constexpr double w_5678 = 1./36;

using Probability_t = Eigen::Matrix<double, 9, 1>;
 
using DensityField_t = Eigen::Array<double, Eigen::Dynamic, 1>;
using VelocityField_t = Eigen::Array<double, Eigen::Dynamic, 1>;
using ProbabilityField_t = Eigen::Array<double, 9, Eigen::Dynamic, Eigen::RowMajor>;

/*
 * Return the equilibrium distribution function.
 *
 * Parameters
 * ----------
 * rho: double
 *     Fluid density.
 * ux: double
 *     x-component of streaming velocity.
 * uy: double
 *     y-component of streaming velocity.
 *
 * Returns
 * -------
 * f_i: Probability_t
 *     Equilibrium distribution for the given fluid density *rho* and
 *     streaming velocity *ux*, *uy*.
 */
Probability_t equilibrium1(double rho, double ux, double uy) {
    double cu5 = ux + uy;
    double cu6 = -ux + uy;
    double cu7 = -ux - uy;
    double cu8 = ux - uy;
    double uu = ux*ux + uy*uy;
    return (Probability_t() <<
        w_0*rho*(1 - 3./2*uu),
        w_1234*rho*(1 + 3*ux + 9./2*ux*ux - 3./2*uu),
        w_1234*rho*(1 + 3*uy + 9./2*uy*uy - 3./2*uu),
        w_1234*rho*(1 - 3*ux + 9./2*ux*ux - 3./2*uu),
        w_1234*rho*(1 - 3*uy + 9./2*uy*uy - 3./2*uu),
        w_5678*rho*(1 + 3*cu5 + 9./2*cu5*cu5 - 3./2*uu),
        w_5678*rho*(1 + 3*cu6 + 9./2*cu6*cu6 - 3./2*uu),
        w_5678*rho*(1 + 3*cu7 + 9./2*cu7*cu7 - 3./2*uu),
        w_5678*rho*(1 + 3*cu8 + 9./2*cu8*cu8 - 3./2*uu)).finished();
}

/*
 * Return the equilibrium distribution function for an array of values.
 *
 * Parameters
 * ----------
 * rho_kl: DensityField_t
 *     Fluid density on the 2D grid.
 * ux_kl: double
 *     x-component of streaming velocity on the 2D grid.
 * uy_kl: double
 *     y-component of streaming velocity on the 2D grid.
 * f_ikl: ProbabilityField_t
 *     Equilibrium distribution for the given fluid density *rho_kl* and
 *     streaming velocity *ux_kl*, *uy_kl* on the same 2D grid.
 */
void equilibriumn(Eigen::Ref<DensityField_t> rho_kl,
                  Eigen::Ref<VelocityField_t> ux_kl,
                  Eigen::Ref<VelocityField_t> uy_kl,
                  Eigen::Ref<ProbabilityField_t> f_ikl) {
    using Stride = Eigen::Stride<1, Eigen::Dynamic>;
    for (int kl = 0; kl < f_ikl.cols(); ++kl) {
        Eigen::Map<Probability_t, Eigen::Unaligned, Stride> f_i(f_ikl.data() + kl, Stride(1, f_ikl.cols()));
        f_i = equilibrium1(rho_kl(kl), ux_kl(kl), uy_kl(kl));
    }
}

/*
 * Carry out collision operation for an array of values.
 *
 * Parameters
 * ----------
 * f_ikl: ProbabilityField_t
 *     Equilibrium distribution for the given fluid density *rho_kl* and
 *     streaming velocity *ux_kl*, *uy_kl* on the same 2D grid.
 * omega: double
 *     Relaxation parameter.
 */
void colliden(Eigen::Ref<ProbabilityField_t> f_ikl, double omega) {
    using Stride = Eigen::Stride<1, Eigen::Dynamic>;
    for (int kl = 0; kl < f_ikl.cols(); ++kl) {
        Eigen::Map<Probability_t, Eigen::Unaligned, Stride> f_i(f_ikl.data() + kl, Stride(1, f_ikl.cols()));
        double rho = f_i.sum();
        double ux = (f_i(1) - f_i(3) + f_i(5) - f_i(6) - f_i(7) + f_i(8))/rho;
        double uy = (f_i(2) - f_i(4) + f_i(5) + f_i(6) - f_i(7) - f_i(8))/rho;
        f_i += omega*(equilibrium1(rho, ux, uy) - f_i);
    }
}

PYBIND11_MODULE(shear_wave_opt2_cext, m) {
    m.doc() = "Lattice Boltzmann kernels";

    m.def("equilibrium", &equilibrium1,
    	  "Return the equilibrium distribution function.");
    m.def("equilibrium", &equilibriumn,
    	  "Return the equilibrium distribution function for an array of values.");
    m.def("collide", &colliden,
    	  "Carry out collision operation for an array of values.");
}
