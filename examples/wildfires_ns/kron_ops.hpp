// SPDX-License-Identifier: MIT

#ifndef WILDFIRES_NS_KRON_OPS_HPP
#define WILDFIRES_NS_KRON_OPS_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "ads/form_matrix.hpp"
#include "ads/lin/band_matrix.hpp"
#include "ads/lin/band_solve.hpp"
#include "ads/lin/solver_ctx.hpp"
#include "ads/lin/tensor.hpp"
#include "ads/simulation/dimension.hpp"
#include "ads/solver.hpp"

namespace ads::kron {

using tensor2 = lin::tensor<double, 2>;

struct dim_op {
    lin::band_matrix full;
    lin::band_matrix hat;
    lin::solver_ctx ctx;

    dim_op(const dimension& d, double cm, double ck, bool fix_ends)
    : full{d.p, d.p, d.dofs()}
    , hat{d.p, d.p, d.dofs()}
    , ctx{hat} {
        const int n = d.dofs();
        const int p = d.p;
        lin::band_matrix M{p, p, n};
        lin::band_matrix K{p, p, n};
        gram_matrix_1d(M, d.basis);
        stiffness_matrix_1d(K, d.basis);
        for (int i = 0; i < n; ++i) {
            for (int j = std::max(0, i - p); j <= std::min(n - 1, i + p); ++j) {
                const double v = cm * M(i, j) + ck * K(i, j);
                full(i, j) = v;
                hat(i, j) = v;
            }
        }
        if (fix_ends) {
            for (int k : {0, n - 1}) {
                for (int j = std::max(0, k - p); j <= std::min(n - 1, k + p); ++j) {
                    hat(k, j) = 0;
                }
                hat(k, k) = 1;
            }
        }
        lin::factorize(hat, ctx);
    }

    dim_op(const dim_op&) = delete;
    dim_op& operator=(const dim_op&) = delete;

    ads::dim_data data() { return {hat, ctx}; }
    int size() const { return full.rows; }
};

inline void apply(const dim_op& A, const dim_op& B, const tensor2& g, tensor2& out) {
    const int nx = g.size(0), ny = g.size(1);
    const int px = A.full.kl, py = B.full.kl;
    tensor2 z{{nx, ny}};
    for (int i = 0; i < nx; ++i) {
        for (int l = 0; l < ny; ++l) {
            double s = 0;
            for (int k = std::max(0, i - px); k <= std::min(nx - 1, i + px); ++k) {
                s += A.full(i, k) * g(k, l);
            }
            z(i, l) = s;
        }
    }
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double s = 0;
            for (int l = std::max(0, j - py); l <= std::min(ny - 1, j + py); ++l) {
                s += B.full(j, l) * z(i, l);
            }
            out(i, j) = s;
        }
    }
}

inline void solve_dirichlet(dim_op& A, dim_op& B, tensor2& rhs, const tensor2& g, tensor2& buf,
                            tensor2& scratch) {
    const int nx = rhs.size(0), ny = rhs.size(1);
    apply(A, B, g, scratch);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const bool bd = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
            rhs(i, j) = bd ? 0.0 : rhs(i, j) - scratch(i, j);
        }
    }
    ads::ads_solve(rhs, buf, A.data(), B.data());
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            rhs(i, j) += g(i, j);
        }
    }
}

namespace detail {

inline void band_solve_raw(dim_op& A, double* b, int ldb, int nrhs) {
    const char* trans = "No transpose";
    int info = 0;
    auto& M = A.hat;
    dgbtrs_(trans, &M.cols, &M.kl, &M.ku, &nrhs, M.full_buffer(), &A.ctx.lda, A.ctx.pivot(), b, &ldb,
            &info);
}

template <typename Exec>
inline bool solve_sweeps_parallel(const Exec& exec, dim_op& A, dim_op& B, tensor2& rhs) {
    const int nx = rhs.size(0), ny = rhs.size(1);
    double* d = &rhs(0, 0);
    if (nx < 2 || ny < 2 || &rhs(1, 0) - d != 1 || &rhs(0, 1) - d != nx) {
        return false;
    }
    constexpr int block = 32;
    {
        std::vector<int> tasks;
        for (int j = 0; j < ny; j += block) {
            tasks.push_back(j);
        }
        exec.for_each(tasks, [&](int j0) {
            const int cnt = std::min(block, ny - j0);
            band_solve_raw(A, d + static_cast<std::size_t>(j0) * nx, nx, cnt);
        });
    }
    {
        std::vector<int> tasks;
        for (int i = 0; i < nx; i += block) {
            tasks.push_back(i);
        }
        exec.for_each(tasks, [&](int i0) {
            const int cnt = std::min(block, nx - i0);
            std::vector<double> tmp(static_cast<std::size_t>(ny) * cnt);
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < cnt; ++k) {
                    tmp[static_cast<std::size_t>(k) * ny + j] = d[i0 + k + static_cast<std::size_t>(j) * nx];
                }
            }
            band_solve_raw(B, tmp.data(), ny, cnt);
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < cnt; ++k) {
                    d[i0 + k + static_cast<std::size_t>(j) * nx] = tmp[static_cast<std::size_t>(k) * ny + j];
                }
            }
        });
    }
    return true;
}

}  // namespace detail

template <typename Exec>
inline void solve_dirichlet_par(const Exec& exec, dim_op& A, dim_op& B, tensor2& rhs,
                                const tensor2& g, tensor2& buf, tensor2& scratch) {
    const int nx = rhs.size(0), ny = rhs.size(1);
    apply(A, B, g, scratch);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const bool bd = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
            rhs(i, j) = bd ? 0.0 : rhs(i, j) - scratch(i, j);
        }
    }
    if (!detail::solve_sweeps_parallel(exec, A, B, rhs)) {
        ads::ads_solve(rhs, buf, A.data(), B.data());
    }
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            rhs(i, j) += g(i, j);
        }
    }
}

template <typename Exec>
inline void solve_free_par(const Exec& exec, dim_op& A, dim_op& B, tensor2& rhs, tensor2& buf) {
    if (!detail::solve_sweeps_parallel(exec, A, B, rhs)) {
        ads::ads_solve(rhs, buf, A.data(), B.data());
    }
}

// Solves (A^-1 (x) B^-1) rhs in place (no boundary conditions: the pressure systems).
inline void solve_free(dim_op& A, dim_op& B, tensor2& rhs, tensor2& buf) {
    ads::ads_solve(rhs, buf, A.data(), B.data());
}

}  // namespace ads::kron

#endif  // WILDFIRES_NS_KRON_OPS_HPP
