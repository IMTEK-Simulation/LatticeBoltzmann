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
| D2Q9 |
--------

C++ kernel containing an implementation of the collision operation on a D2Q9
lattice.
*/

#ifndef __D2Q9_H
#define __D2Q9_H

#include "lbkernels.h"

template<typename T>
using D2Q9Probability_t = Eigen::Matrix<T, 9, 1>;

template<typename T>
using D2Q9ProbabilityField_t = Eigen::Array<T, 9, Eigen::Dynamic, Eigen::RowMajor>;

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
 * f_i: D2Q9Probability_t
 *     Equilibrium distribution for the given fluid density *rho* and
 *     streaming velocity *ux*, *uy*.
 */
template<typename T>
D2Q9Probability_t<T> d2q9_equilibrium1(T rho, T ux, T uy) {
    T w_0 = 4*rho/9;
    T w_1234 = rho/9;
    T w_5678 = rho/36;
    ux *= 3;
    uy *= 3;
    T cu5 = ux + uy;
    T cu6 = -ux + uy;
    T cu7 = -ux - uy;
    T cu8 = ux - uy;
    T uu = (ux*ux + uy*uy)/6;
    return (D2Q9Probability_t<T>() <<
        w_0*(1 - uu),
        w_1234*(1 + ux + ux*ux/2 - uu),
        w_1234*(1 + uy + uy*uy/2 - uu),
        w_1234*(1 - ux + ux*ux/2 - uu),
        w_1234*(1 - uy + uy*uy/2 - uu),
        w_5678*(1 + cu5 + cu5*cu5/2 - uu),
        w_5678*(1 + cu6 + cu6*cu6/2 - uu),
        w_5678*(1 + cu7 + cu7*cu7/2 - uu),
        w_5678*(1 + cu8 + cu8*cu8/2 - uu)).finished();
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
 * f_ikl: D2Q9ProbabilityField_t
 *     Equilibrium distribution for the given fluid density *rho_kl* and
 *     streaming velocity *ux_kl*, *uy_kl* on the same 2D grid.
 */
template<typename T>
void d2q9_equilibriumn(Eigen::Ref<DensityField_t<T>> rho_kl,
                       Eigen::Ref<VelocityField_t<T>> ux_kl,
                       Eigen::Ref<VelocityField_t<T>> uy_kl,
                       Eigen::Ref<D2Q9ProbabilityField_t<T>> f_ikl) {
    using Stride = Eigen::Stride<1, Eigen::Dynamic>;
    for (int kl = 0; kl < f_ikl.cols(); ++kl) {
        Eigen::Map<D2Q9Probability_t<T>, Eigen::Unaligned, Stride> f_i(f_ikl.data() + kl, Stride(1, f_ikl.cols()));
        f_i = d2q9_equilibrium1(rho_kl(kl), ux_kl(kl), uy_kl(kl));
    }
}

/*
 * Carry out collision operation for an array of values.
 *
 * Parameters
 * ----------
 * f_ikl: D2Q9ProbabilityField_t
 *     Equilibrium distribution for the given fluid density *rho_kl* and
 *     streaming velocity *ux_kl*, *uy_kl* on the same 2D grid.
 * omega: double
 *     Relaxation parameter.
 */
template<typename T>
void d2q9_colliden(Eigen::Ref<D2Q9ProbabilityField_t<T>> f_ikl, T omega) {
    using Stride = Eigen::Stride<1, Eigen::Dynamic>;
    for (int kl = 0; kl < f_ikl.cols(); ++kl) {
        Eigen::Map<D2Q9Probability_t<T>, Eigen::Unaligned, Stride> f_i(f_ikl.data() + kl, Stride(1, f_ikl.cols()));
        T rho = f_i.sum();
        T ux = (f_i(1) - f_i(3) + f_i(5) - f_i(6) - f_i(7) + f_i(8))/rho;
        T uy = (f_i(2) - f_i(4) + f_i(5) + f_i(6) - f_i(7) - f_i(8))/rho;
        f_i += omega*(d2q9_equilibrium1(rho, ux, uy) - f_i);
    }
}

#endif