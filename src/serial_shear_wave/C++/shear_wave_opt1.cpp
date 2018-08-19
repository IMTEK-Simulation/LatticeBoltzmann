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
| OPT1 |
--------

This is an implementation of the D2Q9 Lattice Boltzmann lattice in the simple
relaxation time approximation. The code reports the amplitude of a decaying
shear wave which can be used to measure viscosity.

The present implementation contains was optimized with respect to opt1.
The optimization is identical to the opt1 Python code.

This code requires pybind11, Eigen and boost. Compile with
c++ -O3 -Wall -std=c++11 -I/usr/local/Cellar/pybind11/2.2.3/include -I/usr/local/Cellar/eigen/3.3.5/include/eigen3 shear_wave_opt0.cpp -o shear_wave_opt0
*/

#include <boost/range/combine.hpp>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

class D2Q9 {
public:
    using Density_t = double;
    using Velocity_t = double;
    using Probability_t = Eigen::Matrix<double, 9, 1>;

    using ChannelVelocity_t = Eigen::Matrix<int, 2, 9>;
    using Weight_t = Eigen::Array<double, 9, 1>;

    using DensityField_t = std::vector<Density_t>;
    using VelocityField_t = std::vector<Velocity_t>;
    using ProbabilityField_t = std::vector<Probability_t>;

    const ChannelVelocity_t c_ci;

    static constexpr double w_0 = 4./9;
    static constexpr double w_1234 = 1./9;
    static constexpr double w_5678 = 1./36;

    D2Q9(int nx, int ny, double omega):
        c_ci((ChannelVelocity_t() << 0,  1,  0, -1,  0,  1, -1, -1,  1,
                                     0,  0,  1,  0, -1,  1,  1, -1, -1).finished()) {
        this->nx = nx;
        this->ny = ny;
        this->omega = omega;
    }

    Probability_t equilibrium(double rho, Velocity_t ux, Velocity_t uy) {
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

    void equilibrium(DensityField_t &rho_kl,
                     VelocityField_t &ux_kl, VelocityField_t &uy_kl,
                     ProbabilityField_t &f_kli) {
        for (auto && tup: boost::combine(rho_kl, ux_kl, uy_kl, f_kli)) {
            auto && rho = tup.get<0>();
            auto && ux = tup.get<1>();
            auto && uy = tup.get<2>();
            auto && f_i = tup.get<3>();
            f_i = this->equilibrium(rho, ux, uy);
        }
    }

    void equilibrium(double rho, Velocity_t &ux, Velocity_t &uy,
                     ProbabilityField_t &f_kli) {
        for (auto && f_i: f_kli) {
            f_i = this->equilibrium(rho, ux, uy);
        }
    }

    void collide(ProbabilityField_t &f_kli) {
        for (auto && f_i: f_kli) {
            double rho = f_i.sum();
            Velocity_t ux = (f_i(1) - f_i(3) + f_i(5) - f_i(6) - f_i(7) + f_i(8))/rho;
            Velocity_t uy = (f_i(2) - f_i(4) + f_i(5) + f_i(6) - f_i(7) - f_i(8))/rho;
            f_i += this->omega*(equilibrium(rho, ux, uy) - f_i);
        }
    }

    void stream(ProbabilityField_t &f_kli) {
        ProbabilityField_t g_kli(f_kli.size());
        auto f_i = f_kli.begin();
        for (int l = 0; l < this->ny; ++l) {
            for (int k = 0; k < this->nx; ++k, ++f_i) {
                for (int i = 0; i < 9; i++) {
                    int k1 = k + this->c_ci(0, i);
                    while (k1 < 0) k1 += this->nx;
                    while (k1 >= this->nx) k1 -= this->nx;
                    int l1 = l + this->c_ci(1, i);
                    while (l1 < 0) l1 += this->ny;
                    while (l1 >= this->ny) l1 -= this->ny;
                    g_kli[l1*this->nx + k1](i) = (*f_i)(i);
                }
            }
        }
        f_kli = g_kli;
    }
private:
    int nx, ny;
    double omega;
};

int main(int argc, char *argv[])
{
    int nx = 300;
    int ny = 300;
    int nsteps = 1000;
    double omega = 0.3;
    D2Q9 lb(nx, ny, omega);
    D2Q9::ProbabilityField_t f_kli(nx*ny);

    auto f_i = f_kli.begin();
    for (int l = 0; l < ny; ++l) {
        for (int k = 0; k < nx; ++k, ++f_i) {
            D2Q9::Velocity_t uy = std::sin(2*M_PI/nx*k);
            *f_i = lb.equilibrium(1.0, 0.0, uy);
        }
    }

    for (int n = 0; n < nsteps; ++n) {
        lb.stream(f_kli);
        lb.collide(f_kli);

        double accum = 0.0;
        for (int k = 0; k < nx; ++k)
            accum += (lb.c_ci.cast<double>()*f_kli[ny/2*nx+k])(1)*std::sin(2*M_PI/nx*k);
        std::cout << accum*2/nx << std::endl;
    }
}
