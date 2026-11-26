// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_SOLVER_MUMPS_HPP
#define ADS_SOLVER_MUMPS_HPP

#include "ads/config.hpp"

#ifdef ADS_USE_MUMPS

#    include <dmumps_c.h>
#    include <mpi.h>

#    include <chrono>
#    include <cstdint>
#    include <cstdio>
#    include <iostream>
#    include <vector>

#    include "ads/util.hpp"

namespace ads::mumps {

struct problem {
    problem(double* rhs, int n)
    : rhs_{rhs}
    , n{n} { }

    explicit problem(std::vector<double>& rhs)
    : rhs_{rhs.data()}
    , n{static_cast<int>(rhs.size())} { }

    void add(int row, int col, double value) {
        rows_.push_back(row);
        cols_.push_back(col);
        values_.push_back(value);
    }

    int nonzero_entries() const { return narrow_cast<int>(values_.size()); }

    void reserve(int entries) {
        rows_.reserve(entries);
        cols_.reserve(entries);
        values_.reserve(entries);
    }

    int dofs() const { return n; }

    int* irn() { return rows_.data(); }

    int* jcn() { return cols_.data(); }

    double* a() { return values_.data(); }

    double* rhs() { return rhs_; }

    void rhs(double* data) { rhs_ = data; }

private:
    std::vector<int> rows_;
    std::vector<int> cols_;
    std::vector<double> values_;
    double* rhs_;
    int n;
};

struct phase_timers {
    double analyze_s = 0, factorize_s = 0, solve_s = 0;
    long calls = 0;
};

class solver {
private:
    DMUMPS_STRUC_C id{};
    phase_timers timers_;

    int& icntl(int idx) { return id.icntl[idx - 1]; }

    double& cntl(int idx) { return id.cntl[idx - 1]; }

    int info(int idx) const { return id.info[idx - 1]; }

    double rinfo(int idx) const { return id.rinfo[idx - 1]; }

    int infog(int idx) const { return id.infog[idx - 1]; }

    double rinfog(int idx) const { return id.rinfog[idx - 1]; }

public:
    solver() {
        int mpi_up = 0;
        MPI_Initialized(&mpi_up);
        if (!mpi_up) {
            int argc = 0;
            char** argv = nullptr;
            MPI_Init(&argc, &argv);
        }
        ++instances_();

        id.job = -1;
        id.par = 1;
        id.sym = 0;
        id.comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);

        dmumps_c(&id);

        // Use ordering scheme: auto (0), sequential (1), parallel (2)
        icntl(28) = 1;

        // Sequential orderings:
        // Metis (5), PORD (4), SCOTCH (3), AMD (0), AMF (2), QAMD (6), automatic choice (7)
        icntl(7) = 7;

        // Parallel orderings:
        // PT-SCOTCH (1), ParMetis (2), automatic choice (0)
        // icntl(29) = 2;

        // extra space factor
        icntl(14) = 200;

        // null pivot detection
        //icntl(24) = 1;
        cntl(3) = 1e-5;

        // out-of-core
        // icntl(22) = 1;

        // streams
        icntl(1) = 3;
        icntl(2) = 3;
        icntl(3) = 3;
        icntl(4) = 3;
    }

    solver(const solver&) = delete;
    solver& operator=(const solver&) = delete;
    solver(solver&&) = delete;
    solver& operator=(solver&&) = delete;

    void print_state(const char* text) const {
        if (id.info[0] != 0) {
            std::cout << text << ":" << std::endl;
            std::cout << "  INFO(1) = " << id.info[0] << std::endl;
            std::cout << "  INFO(2) = " << id.info[1] << std::endl;
            std::cout << "Error: ";
            report_error(std::cout, id.info[0], id.info[1]);
            std::cout << std::endl;
        }
    }

    void save_to_file(problem& problem, const char* output_path) {
        prepare_(problem);
        set_output_path(output_path);

        analyze_();
    }

    const phase_timers& timers() const { return timers_; }

    void solve(problem& problem, const char* output_path = nullptr) {
        using clk = std::chrono::steady_clock;
        auto sec = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };

        prepare_(problem);

        if (output_path != nullptr) {
            set_output_path(output_path);
        }

        auto t0 = clk::now();
        analyze_();
        // std::cout << "Analysis type: " << infog(32) << std::endl;
        // std::cout << "Ordering used: " << infog(7) << std::endl;
        // report_after_analysis(std::cout);
        auto t1 = clk::now();
        factorize_();
        // std::cout << "Deficiency: " << infog(28) << std::endl;
        auto t2 = clk::now();
        solve_();
        auto t3 = clk::now();
        timers_.analyze_s += sec(t0, t1);
        timers_.factorize_s += sec(t1, t2);
        timers_.solve_s += sec(t2, t3);
        ++timers_.calls;
    }

    void factorize(problem& problem) {
        using clk = std::chrono::steady_clock;
        prepare_(problem);
        auto t0 = clk::now();
        analyze_();
        auto t1 = clk::now();
        factorize_();
        auto t2 = clk::now();
        timers_.analyze_s += std::chrono::duration<double>(t1 - t0).count();
        timers_.factorize_s += std::chrono::duration<double>(t2 - t1).count();
    }

    void solve_factored(double* rhs) {
        auto t0 = std::chrono::steady_clock::now();
        id.rhs = rhs;
        id.nrhs = 1;
        id.lrhs = id.n;
        solve_();
        timers_.solve_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++timers_.calls;
    }

    double flops_assembly() const { return rinfog(2); }

    double flops_elimination() const { return rinfog(3); }

    ~solver() {
        id.job = -2;
        dmumps_c(&id);
        if (--instances_() == 0) {
            MPI_Finalize();
        }
    }

private:
    static int& instances_() {
        static int n = 0;
        return n;
    }

    void prepare_(problem& problem) {
        id.n = problem.dofs();
        id.nz = problem.nonzero_entries();

        id.irn = problem.irn();
        id.jcn = problem.jcn();
        id.a = problem.a();

        id.rhs = problem.rhs();
        id.nrhs = 1;
        id.lrhs = id.n;
    }

    void factorize_() {
        constexpr int max_retries = 5;
        for (int attempt = 0;; ++attempt) {
            id.job = 2;
            dmumps_c(&id);
            if (id.info[0] != -9 || attempt >= max_retries) {
                break;
            }
            const int old = icntl(14);
            icntl(14) = old * 2;
            std::cout << "[mumps] INFO(1) = -9 (workspace too small, missing " << id.info[1]
                      << "); ICNTL(14) " << old << " -> " << icntl(14) << ", repeating factorization"
                      << std::endl;
        }
        print_state("After factorize_()");
    }

    void analyze_() {
        id.job = 1;
        dmumps_c(&id);
        print_state("After analyze_()");
    }

    void solve_() {
        id.job = 3;
        dmumps_c(&id);
        print_state("After solve_()");
    }

    void set_output_path(const char* output_path) {
        constexpr auto size = sizeof(id.write_problem);
        std::snprintf(id.write_problem, size, "%s", output_path);
    }

    void report_error(std::ostream& os, int info1, int info2) const {
        switch (info1) {
        case -6: os << "Matrix singular in structure (rank = " << info2 << ")"; break;
        case -7: os << "Problem of integer workspace allocation (size = " << info2 << ")"; break;
        case -8: os << "Main internal integer workarray is too small"; break;
        case -9:
            os << "Main internal real/complex workarray is too small (missing = " << info2 << ")";
            break;
        case -10: os << "Numerically singular matrix"; break;
        case -13: os << "Problem of workspace allocation (size = " << info2 << ")"; break;
        default: os << "Problem code: " << info1;
        }
    }

    std::int64_t handle_neg(int value) const {
        if (value >= 0) {
            return value;
        } else {
            return -value * 1'000'000L;
        }
    }

    double as_MB(std::int64_t size) const {
        return static_cast<double>(size) / (1024 * 1024) * sizeof(double);
    }

    void report_after_analysis(std::ostream& os) const {
        os << "MUMPS (" << id.version_number << ") after analysis:" << std::endl;
        os << "RINFO:" << std::endl;
        os << "  Estimated FLOPS for the elimination:          " << rinfo(1) << std::endl;
        os << "  Disk space for out-of-core factorization:     " << rinfo(5) << " MB" << std::endl;
        os << "  Size of the file used to save data:           " << rinfo(7) << " MB" << std::endl;
        os << "  Size of the MUMPS structure:                  " << rinfo(8) << " MB" << std::endl;
        os << "INFO:" << std::endl;
        os << "  Success:                                      " << info(1) << std::endl;
        auto real_store = handle_neg(info(3));
        os << "  Size of the real space to store factors:      " << real_store << " ("
           << as_MB(real_store) << " MB)" << std::endl;
        os << "  Size of the integer space to store factors:   " << info(4) << std::endl;
        os << "  Estimated maximum front size:                 " << info(5) << std::endl;
        os << "  Number of nodes in a tree:                    " << info(6) << std::endl;
        os << "  Size of the integer space to factorize:       " << info(7) << std::endl;
        auto real_factor = handle_neg(info(8));
        os << "  Size of the real space to factorize:          " << real_factor << " ("
           << as_MB(real_factor) << " MB)" << std::endl;
        os << "  Total memory needed:                          " << info(15) << " MB" << std::endl;
        os << "  Total memory needed (OoC):                    " << info(17) << " MB" << std::endl;
        os << "  Size of the integer space to factorize (OoC): " << info(19) << std::endl;
        auto real_factor_ooc = handle_neg(info(20));
        os << "  Size of the real space to factorize (OoC):    " << real_factor_ooc << " ("
           << as_MB(real_factor_ooc) << " MB)" << std::endl;
        os << "  Estimated number of entries in factors:       " << handle_neg(info(24))
           << std::endl;
        auto low_real_factor = handle_neg(info(29));
        os << "  Size of the real space to factorize (low-r):  " << low_real_factor << " ("
           << as_MB(low_real_factor) << " MB)" << std::endl;
        os << "  Total memory needed (low-rank):               " << info(30) << " MB" << std::endl;
        os << "  Total memory needed (low-rank, OoC):          " << info(31) << " MB" << std::endl;
    }
};

}  // namespace ads::mumps

#endif  // defined(ADS_USE_MUMPS)

#endif  // ADS_SOLVER_MUMPS_HPP
