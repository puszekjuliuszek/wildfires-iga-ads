// SPDX-License-Identifier: MIT

#ifndef WILDFIRES_NS_NS_INPROC_HPP
#define WILDFIRES_NS_NS_INPROC_HPP

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ads/bspline/bspline.hpp"
#include "wildfires_navier_stokes_2d.hpp"

namespace ads {

class ns_inproc {
public:
    using ns_type = wildfires_navier_stokes_2d<prob_pyrogenic_plume>;

    struct field_grid {
        int nx = 0, ny = 0;
        std::vector<double> T;

        double operator()(point_type p) const {
            double fx = p[0] * (nx - 1);
            double fy = p[1] * (ny - 1);
            int ix = std::max(0, std::min(nx - 2, static_cast<int>(fx)));
            int iy = std::max(0, std::min(ny - 2, static_cast<int>(fy)));
            double tx = fx - ix;
            double ty = fy - iy;
            double t00 = T[ix * ny + iy];
            double t10 = T[(ix + 1) * ny + iy];
            double t01 = T[ix * ny + iy + 1];
            double t11 = T[(ix + 1) * ny + (iy + 1)];
            return t00 * (1 - tx) * (1 - ty) + t10 * tx * (1 - ty) + t01 * (1 - tx) * ty
                 + t11 * tx * ty;
        }
    };

    struct params {
        int Nx = 0, Ny = 0;
        double Lx = 1, Ly = 1;
        int nsteps = 50;
        double t_end = 2.0;
        double kb = 0, h = 0, Re = 0;
        double wind_x_mps = 0, wind_y_mps = 0, U_mps = 1000.0;
        bool normalize_sink = true;
        bool farfield = true;
        bool warm = true;
        bool factor_cache = false;
        bool kron = true;
        int diag_every = 1;
        int out_nx = 21, out_ny = 21;
        int fields_nx = 200, fields_ny = 200;
        std::string save_prefix;
        std::string diag_file;
        int cycle = 0;
        bool timing = false;
        bool rot_diag = false;
    };

    struct result {
        std::vector<double> u, v;
        double wall_s = 0;
    };

    ns_inproc(galois_executor& executor, int host_threads, int ns_threads)
    : executor{executor}
    , host_threads{host_threads}
    , ns_threads{ns_threads} { }

    void forget_state() {
        if (sim) {
            sim->discard_state();
        }
    }

    result solve(const field_grid& grid, const params& prm) {
        using clk = std::chrono::steady_clock;
        const auto t_main0 = clk::now();
        auto t_lap = t_main0;
        auto lap = [&t_lap]() {
            auto now = clk::now();
            double d = std::chrono::duration<double>(now - t_lap).count();
            t_lap = now;
            return d;
        };

        executor.thread_count(ns_threads);
        struct restore {
            galois_executor& ex;
            int n;
            ~restore() { ex.thread_count(n); }
        } restore_threads{executor, host_threads};

        prob_pyrogenic_plume problem{prm.Re};
        problem.navier_stokes = true;
        problem.kb = prm.kb;
        problem.h = prm.h;
        problem.wind_x = prm.wind_x_mps / prm.U_mps;
        problem.wind_y = prm.wind_y_mps / prm.U_mps;
        problem.input_is_rate = true;
        const double Lx = prm.Lx, Ly = prm.Ly;
        problem.temperature_field = [grid, Lx, Ly](point_type pt) {
            return grid({pt[0] / Lx, pt[1] / Ly});
        };

        if (!sim) {
            const int p = 2, c = 1, quad = p + 1, ders = 2;
            auto mk = [&](int n, double b) {
                int rep = p - 1 - c;
                auto basis = bspline::create_basis(0.0, b, p, n, rep);
                return dimension{basis, quad, ders, 1};
            };
            auto U1x = mk(prm.Nx, Lx), U1y = mk(prm.Ny, Ly);
            auto U2x = mk(prm.Nx, Lx), U2y = mk(prm.Ny, Ly);
            auto make_p = [&](int n, double b) {
                auto basis = bspline::create_basis(0.0, b, 1, n, 0);
                return dimension{basis, quad, ders, 1};
            };
            auto Px = make_p(prm.Nx, Lx), Py = make_p(prm.Ny, Ly);
            space_set trial{U1x, U1y, U2x, U2y, Px, Py};
            space_set test{U1x, U1y, U2x, U2y, Px, Py};
            timesteps_config steps{prm.nsteps, prm.t_end / prm.nsteps};
            sim = std::make_unique<ns_type>(trial, test, steps, problem, prm.fields_nx,
                                            prm.fields_ny, &executor);
            sim->set_save_every_step(false);
            sim->set_factor_cache(prm.factor_cache);
            sim->set_kron_solver(prm.kron);
            sim->set_diag_every(prm.diag_every);
            Nx = prm.Nx;
            Ny = prm.Ny;
            this->Lx = Lx;
            this->Ly = Ly;
        } else if (Nx != prm.Nx || Ny != prm.Ny || this->Lx != Lx || this->Ly != Ly) {
            throw std::runtime_error("ns_inproc: the NS mesh/domain changed between cycles");
        }
        sim->get_problem() = problem;
        const double tm_setup = lap();

        if (prm.normalize_sink) {
            sim->normalize_sink(2 * (grid.nx - 1), 2 * (grid.ny - 1));
        }
        const double tm_sink_norm = lap();
        if (prm.farfield) {
            sim->enable_farfield_bc();
        }
        const double tm_farfield = lap();

        const auto tm0 = sim->timing;
        const auto mt0 = sim->mumps_timers();
        auto t0 = clk::now();
        sim->run_cycle(prm.nsteps, prm.t_end / prm.nsteps, prm.warm);
        result res;
        res.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
        lap();

        res.u.resize(static_cast<std::size_t>(prm.out_nx) * prm.out_ny);
        res.v.resize(res.u.size());
        for (int i = 0; i < prm.out_nx; ++i) {
            for (int j = 0; j < prm.out_ny; ++j) {
                double xbar = static_cast<double>(i) / (prm.out_nx - 1);
                double ybar = static_cast<double>(j) / (prm.out_ny - 1);
                auto [u, v] = sim->sample_velocity(xbar * Lx, ybar * Ly);
                res.u[i * prm.out_ny + j] = u;
                res.v[i * prm.out_ny + j] = v;
            }
        }
        const double tm_write_out = lap();

        auto mb = sim->compute_mass_balance(0.0);
        const double tm_mass_balance = lap();
        std::cout << "[ns_inproc] cycle=" << prm.cycle << " nsteps=" << prm.nsteps
                  << " warm=" << (sim->was_warm_started() ? 1 : 0) << " int_S=" << mb.int_S
                  << " int_div=" << mb.int_div << " flux=" << mb.flux << " |v|max=" << mb.v_max
                  << " wall_run_s=" << res.wall_s << std::endl;
        if (!prm.diag_file.empty()) {
            std::ofstream dfile(prm.diag_file, std::ios::app);
            dfile.precision(10);
            dfile << prm.cycle << " " << mb.int_S << " " << mb.int_div << " " << mb.flux << " "
                  << mb.v_max << " " << mb.v_L2 << " " << (sim->was_warm_started() ? 1 : 0) << " "
                  << res.wall_s << " " << mb.int_S_x << " " << mb.int_S_y;
            if (prm.rot_diag) {
                auto rd = sim->compute_rotation_diag();
                dfile << " " << rd.omega_L2 << " " << rd.div_L2 << " " << rd.S_L2 << " "
                      << rd.divres_L2 << " " << (rd.S_L2 > 0 ? rd.divres_L2 / rd.S_L2 : 0.0) << " "
                      << rd.vrot_L2 << " " << rd.vmb_L2 << " "
                      << (rd.div_L2 > 0 ? rd.omega_L2 / rd.div_L2 : 0.0) << " "
                      << (rd.vmb_L2 > 0 ? rd.vrot_L2 / rd.vmb_L2 : 0.0);
            }
            dfile << "\n";
        }
        const double tm_diag = lap();
        if (!prm.save_prefix.empty()) {
            sim->save_final(prm.save_prefix, prm.fields_nx, prm.fields_ny);
        }
        const double tm_save_fields = lap();

        if (prm.timing) {
            const auto& tm = sim->timing;
            const auto mt = sim->mumps_timers();
            const char* hdr =
                "cycle,nsteps,Nx,Ny,warm,read_input,setup,sink_norm,farfield,run,before,load_state,"
                "steps,assemble,rhs,mumps_analyze,mumps_factorize,mumps_solve,mumps_calls,"
                "after_step,write_out,mass_balance,diag,save_fields,save_state,main_total,"
                "kron_solve,rhs_v1,rhs_bc,rhs_v2,rhs_p,shift\n";
            bool need_header = !std::ifstream("ns_timing.csv").good();
            std::ofstream tf("ns_timing.csv", std::ios::app);
            if (need_header) {
                tf << hdr;
            }
            tf.precision(6);
            tf << prm.cycle << "," << prm.nsteps << "," << prm.Nx << "," << prm.Ny << ","
               << (sim->was_warm_started() ? 1 : 0) << ",0," << tm_setup << "," << tm_sink_norm
               << "," << tm_farfield << "," << res.wall_s << "," << tm.before - tm0.before << ","
               << tm.load_state - tm0.load_state << "," << tm.steps - tm0.steps << ","
               << tm.assemble - tm0.assemble << "," << tm.rhs - tm0.rhs << ","
               << mt.analyze_s - mt0.analyze_s << "," << mt.factorize_s - mt0.factorize_s << ","
               << mt.solve_s - mt0.solve_s << "," << mt.calls - mt0.calls << ","
               << tm.after_step - tm0.after_step << "," << tm_write_out << "," << tm_mass_balance
               << "," << tm_diag << "," << tm_save_fields << ",0,"
               << std::chrono::duration<double>(clk::now() - t_main0).count() << ","
               << tm.kron - tm0.kron << "," << tm.rhs_v1 - tm0.rhs_v1 << ","
               << tm.rhs_bc - tm0.rhs_bc << "," << tm.rhs_v2 - tm0.rhs_v2 << ","
               << tm.rhs_p - tm0.rhs_p << "," << tm.shift - tm0.shift << "\n";
        }
        return res;
    }

private:
    galois_executor& executor;
    int host_threads, ns_threads;
    std::unique_ptr<ns_type> sim;
    int Nx = 0, Ny = 0;
    double Lx = 0, Ly = 0;
};

}

#endif
