// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <random>

#include "ads/bspline/bspline.hpp"
#include "ads/executor/galois.hpp"
#include "ads/lin/dense_solve.hpp"
#include "kron_ops.hpp"

using namespace ads;
using kron::tensor2;

static dimension make_dim(int n, double len, int p, int c) {
    auto basis = bspline::create_basis(0.0, len, p, n, p - 1 - c);
    return dimension{basis, p + 1, 2, 1};
}

static lin::dense_matrix dense_kron(const lin::band_matrix& A, const lin::band_matrix& B) {
    const int nx = A.rows, ny = B.rows;
    lin::dense_matrix D(nx * ny, nx * ny);
    D.zero();
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nx; ++k)
                for (int l = 0; l < ny; ++l)
                    D(i * ny + j, k * ny + l) = A(i, k) * B(j, l);
    return D;
}

int main() {
    ads::galois_executor exec{4};
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    double worst = 0;

    struct cfg {
        int nx, ny, p, c;
        double lx, ly;
    };
    const cfg cfgs[] = {{5, 8, 2, 1, 1.0, 2.0}, {9, 4, 2, 1, 3.0, 0.5}, {6, 7, 1, 0, 1.0, 1.0},
                        {12, 5, 1, 0, 2.0, 1.0}, {45, 40, 2, 1, 1.0, 1.0}};
    const double cs[] = {1e-5, 1e-2, 1.0};

    for (auto& cf : cfgs) {
        auto dx = make_dim(cf.nx, cf.lx, cf.p, cf.c);
        auto dy = make_dim(cf.ny, cf.ly, cf.p, cf.c);
        const int nx = dx.dofs(), ny = dy.dofs();
        const int N = nx * ny;

        for (double c : cs) {
            for (int half = 0; half < 2; ++half) {
                kron::dim_op Ax{dx, 1.0, half == 0 ? c : 0.0, true};
                kron::dim_op By{dy, 1.0, half == 0 ? 0.0 : c, true};

                tensor2 rhs{{nx, ny}}, g{{nx, ny}}, buf{{nx, ny}}, scr{{nx, ny}};
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j) {
                        rhs(i, j) = U(rng);
                        const bool bd = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
                        g(i, j) = bd ? U(rng) : 0.0;
                    }
                auto D = dense_kron(Ax.full, By.full);
                lin::dense_matrix b(N, 1);
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j) {
                        const bool bd = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
                        b(i * ny + j, 0) = bd ? g(i, j) : rhs(i, j);
                        if (bd) {
                            for (int m = 0; m < N; ++m) D(i * ny + j, m) = 0;
                            D(i * ny + j, i * ny + j) = 1;
                        }
                    }
                lin::solver_ctx ctxd{D};
                lin::factorize(D, ctxd);
                lin::solve_with_factorized(D, b, ctxd);

                tensor2 rhs_par{{nx, ny}};
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j) rhs_par(i, j) = rhs(i, j);
                kron::solve_dirichlet(Ax, By, rhs, g, buf, scr);
                kron::solve_dirichlet_par(exec, Ax, By, rhs_par, g, buf, scr);
                double num = 0, den = 0;
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j) {
                        num = std::max(num, std::abs(rhs(i, j) - b(i * ny + j, 0)));
                        den = std::max(den, std::abs(b(i * ny + j, 0)));
                    }
                double nump = 0;
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        nump = std::max(nump, std::abs(rhs_par(i, j) - b(i * ny + j, 0)));
                worst = std::max(worst, nump / den);
                const double rel = num / den;
                worst = std::max(worst, rel);
                std::cout << "dirichlet nx=" << nx << " ny=" << ny << " p=" << cf.p << " c=" << c
                          << " half=" << half << " rel=" << rel << "\n";
            }
        }

        {
            kron::dim_op Ax{dx, 1.0, 1.0, false}, Ay{dy, 1.0, 1.0, false};
            kron::dim_op Mx{dx, 1.0, 0.0, false}, My{dy, 1.0, 0.0, false};
            tensor2 r{{nx, ny}}, buf{{nx, ny}};
            lin::dense_matrix b(N, 1);
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j) {
                    r(i, j) = U(rng);
                    b(i * ny + j, 0) = r(i, j);
                }
            auto D1 = dense_kron(Ax.full, My.full);
            auto D2 = dense_kron(Mx.full, Ay.full);
            auto DM = dense_kron(Mx.full, My.full);
            lin::solver_ctx c1{D1}, c2{D2};
            lin::factorize(D1, c1);
            lin::factorize(D2, c2);
            lin::solve_with_factorized(D1, b, c1);  // psi
            lin::dense_matrix b2(N, 1);
            for (int m = 0; m < N; ++m) {
                double s = 0;
                for (int k = 0; k < N; ++k) s += DM(m, k) * b(k, 0);
                b2(m, 0) = s;
            }
            lin::solve_with_factorized(D2, b2, c2);  // phi

            tensor2 r_par{{nx, ny}};
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j) r_par(i, j) = r(i, j);
            kron::solve_free(Ax, Ay, r, buf);
            kron::solve_free_par(exec, Ax, Ay, r_par, buf);
            double num = 0, den = 0;
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j) {
                    num = std::max(num, std::abs(r(i, j) - b2(i * ny + j, 0)));
                    den = std::max(den, std::abs(b2(i * ny + j, 0)));
                }
            double nump = 0;
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    nump = std::max(nump, std::abs(r_par(i, j) - b2(i * ny + j, 0)));
            worst = std::max(worst, nump / den);
            const double rel = num / den;
            worst = std::max(worst, rel);
            std::cout << "pressure  nx=" << nx << " ny=" << ny << " p=" << cf.p << " rel=" << rel
                      << "\n";
        }
    }
    std::cout << "worst relative error " << worst << (worst <= 1e-12 ? "  PASS" : "  FAIL") << "\n";
    return worst <= 1e-12 ? 0 : 1;
}
