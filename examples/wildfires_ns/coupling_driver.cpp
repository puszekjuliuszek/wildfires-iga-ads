// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ads/bspline/bspline.hpp"
#include "wildfires_navier_stokes_2d.hpp"

namespace {

ads::dimension make_dimension(int p, int c, int n, int quad, int ders, double a = 0.0,
                              double b = 1.0) {
    int rep = p - 1 - c;
    auto basis = ads::bspline::create_basis(a, b, p, n, rep);
    return ads::dimension{basis, quad, ders, 1};
}

struct FieldGrid {
    int nx = 0, ny = 0;
    std::vector<double> T;


    double operator()(ads::point_type p) const {
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

struct options {
    std::map<std::string, std::string> kv;

    bool has(const std::string& k) const { return kv.count(k) > 0; }
    std::string str(const std::string& k, const std::string& def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atof(it->second.c_str());
    }
    int integer(const std::string& k, int def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::atoi(it->second.c_str());
    }
};

}

int main(int argc, char* argv[]) {
    using clk = std::chrono::steady_clock;
    const auto t_main0 = clk::now();
    auto lap = [](clk::time_point& t) {
        auto now = clk::now();
        double d = std::chrono::duration<double>(now - t).count();
        t = now;
        return d;
    };
    auto t_lap = t_main0;
    std::vector<std::string> pos;
    options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            auto eq = a.find('=');
            if (eq == std::string::npos) {
                opt.kv[a.substr(2)] = "1";
            } else {
                opt.kv[a.substr(2, eq - 2)] = a.substr(eq + 1);
            }
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 7) {
        std::cerr << "Usage: coupling_driver <in.txt> <out.txt> <N> <nsteps> <kb> <h> <Re> "
                     "[--option=value ...]  (see the header comment of coupling_driver.cpp)\n";
        return 1;
    }
    std::string infile = pos[0];
    std::string outfile = pos[1];
    int Nx = std::atoi(pos[2].c_str());
    int nsteps = std::atoi(pos[3].c_str());
    double kb = std::atof(pos[4].c_str());
    double h = std::atof(pos[5].c_str());
    double Re = std::atof(pos[6].c_str());

    int Ny = opt.integer("Ny", Nx);
    double Lx = opt.num("domain-x", 1.0);
    double Ly = opt.num("domain-y", 1.0);
    double t_end = opt.num("t-end", 2.0);
    double U_mps = opt.num("U", 1000.0);
    double wind_x = opt.num("wind-x", 0.0) / U_mps;
    double wind_y = opt.num("wind-y", 0.0) / U_mps;
    std::string input_kind = opt.str("input", "temperature");
    std::string bc_kind = opt.str("bc", "uniform");
    int out_nx = opt.integer("out-nx", 21);
    int out_ny = opt.integer("out-ny", 21);
    int fields_nx = opt.integer("fields-nx", 200);
    int fields_ny = opt.integer("fields-ny", 200);
    bool save_steps = opt.integer("save-steps", 0) != 0;
    if (input_kind != "temperature" && input_kind != "rate") {
        std::cerr << "coupling_driver: --input must be temperature or rate\n";
        return 1;
    }
    if (bc_kind != "uniform" && bc_kind != "farfield") {
        std::cerr << "coupling_driver: --bc must be uniform or farfield\n";
        return 1;
    }

    FieldGrid grid;
    {
        std::ifstream fin(infile);
        if (!fin) {
            std::cerr << "coupling_driver: could not open input file " << infile << "\n";
            return 1;
        }
        std::string header;
        std::getline(fin, header);
        std::istringstream hs(header);
        hs >> grid.nx;
        if (!(hs >> grid.ny)) {
            grid.ny = grid.nx;
        }
        grid.T.resize(static_cast<std::size_t>(grid.nx) * grid.ny);
        for (int i = 0; i < grid.nx; ++i) {
            for (int j = 0; j < grid.ny; ++j) {
                double xbar, ybar, Tv;
                fin >> xbar >> ybar >> Tv;
                grid.T[i * grid.ny + j] = Tv;
            }
        }
    }

    const double tm_read_input = lap(t_lap);

    int p = 2, c = 1;
    int quad = p + 1;
    int ders = 2;

    auto U1x = make_dimension(p, c, Nx, quad, ders, 0.0, Lx);
    auto U1y = make_dimension(p, c, Ny, quad, ders, 0.0, Ly);
    auto U2x = make_dimension(p, c, Nx, quad, ders, 0.0, Lx);
    auto U2y = make_dimension(p, c, Ny, quad, ders, 0.0, Ly);
    auto Px = make_dimension(1, 0, Nx, quad, ders, 0.0, Lx);
    auto Py = make_dimension(1, 0, Ny, quad, ders, 0.0, Ly);

    ads::space_set trial{U1x, U1y, U2x, U2y, Px, Py};
    ads::space_set test{U1x, U1y, U2x, U2y, Px, Py};

    double dt = t_end / nsteps;
    ads::timesteps_config steps{nsteps, dt};

    ads::prob_pyrogenic_plume problem{Re};
    problem.navier_stokes = true;
    problem.kb = kb;
    problem.h = h;
    problem.wind_x = wind_x;
    problem.wind_y = wind_y;
    problem.input_is_rate = (input_kind == "rate");
    auto field = [grid, Lx, Ly](ads::point_type pt) { return grid({pt[0] / Lx, pt[1] / Ly}); };
    problem.temperature_field = field;
    if (opt.has("no-sink")) {
        problem.sink_enabled = false;
    }
    if (opt.has("force-alpha")) {
        if (problem.input_is_rate) {
            std::cerr << "coupling_driver: --force-alpha needs --input=temperature\n";
            return 1;
        }
        problem.force_alpha = opt.num("force-alpha", 0.0);
        problem.sink_enabled = false;
        double hx = 0.5 * Lx / (grid.nx - 1);
        double hy = 0.5 * Ly / (grid.ny - 1);
        problem.temperature_gradient = [field, hx, hy](ads::point_type pt) {
            return ads::point_type{
                (field({pt[0] + hx, pt[1]}) - field({pt[0] - hx, pt[1]})) / (2 * hx),
                (field({pt[0], pt[1] + hy}) - field({pt[0], pt[1] - hy})) / (2 * hy)};
        };
    }

    int cycle = opt.integer("cycle", 0);
    std::cout << "[coupling_driver] cycle=" << cycle << " Nx=" << Nx << " Ny=" << Ny
              << " domain=" << Lx << "x" << Ly << " nsteps=" << nsteps << " t_end=" << t_end
              << " kb=" << kb << " h=" << h << " Re=" << Re << " input=" << input_kind
              << " b=(" << wind_x * U_mps << "," << wind_y * U_mps << ") m/s bc=" << bc_kind
              << std::endl;

    ads::wildfires_navier_stokes_2d<ads::prob_pyrogenic_plume> sim{trial, test, steps, problem,
                                                                   fields_nx, fields_ny};
    const double tm_setup = lap(t_lap);
    sim.set_save_every_step(save_steps);
    if (opt.has("factor-cache")) {
        sim.set_factor_cache(true);
    }
    sim.set_diag_every(opt.integer("diag-every", 1));
    if (opt.str("solver", "kron") == "kron") {
        sim.set_kron_solver(true);
        if (opt.has("kron-generic")) {
            sim.set_kron_fast(false);
        }
    } else if (opt.str("solver", "kron") != "mumps") {
        std::cerr << "[coupling_driver] unknown --solver=" << opt.str("solver", "") << std::endl;
        return 2;
    }
    std::string state_file = opt.str("state", "");
    if (opt.has("warm") && !state_file.empty()) {
        sim.set_warm_start(state_file);
    }
    if (opt.integer("normalize-sink", 0) != 0 && problem.sink_enabled) {
        sim.normalize_sink(2 * (grid.nx - 1), 2 * (grid.ny - 1));
    }
    const double tm_sink_norm = lap(t_lap);
    if (bc_kind == "farfield") {
        sim.enable_farfield_bc();
    }
    const double tm_farfield = lap(t_lap);

    if (opt.has("verify-kron")) {
        double worst_rhs = 0, worst_step = 0, worst_kres = 0, worst_krep = 0, mumps_noise = 1e-12;
        std::string seq = opt.str("verify-kron", "");
        if (seq.empty() || seq == "1") {
            seq = "01";
        }
        for (char ch : seq) {
            const int forced = ch == '1' ? 1 : 0;
            auto r = sim.verify_kron(forced == 1);
            std::cout << "[verify-kron] mode=" << (forced ? "forced-parallel" : "default")
                      << " warm=" << (sim.was_warm_started() ? 1 : 0) << " Nx=" << Nx << " Ny=" << Ny
                      << " rhs: v1=" << r.rhs_v1 << " v2=" << r.rhs_v2 << " p=" << r.rhs_p
                      << " | step vs MUMPS: vx=" << r.step_vx << " vy=" << r.step_vy
                      << " p=" << r.step_p << " phi=" << r.step_phi << " | pressure residual: kron=" << r.res_kron << " mumps=" << r.res_mumps
                      << " | generic rhs repeat=" << r.rep_generic_rhs
                      << " | repeat: kron=" << r.rep_kron
                      << " mumps=" << r.rep_mumps << " (vx " << r.rep_m_vx << ", vy " << r.rep_m_vy << ", p "
                      << r.rep_m_p << ", phi " << r.rep_m_phi << ")" << std::endl;
            worst_rhs = std::max(worst_rhs, r.worst_rhs());
            worst_step = std::max(worst_step, r.worst_step());
            worst_kres = std::max(worst_kres, r.res_kron);
            worst_krep = std::max(worst_krep, r.rep_kron);
            mumps_noise = std::max({mumps_noise, r.res_mumps, r.rep_mumps});
        }
        const double step_limit = 1e-5;
        const bool ok = worst_rhs <= 1e-13 && worst_kres <= 1e-11 && worst_krep <= 1e-13
                        && worst_step <= step_limit && std::isfinite(worst_step);
        std::cout << "[verify-kron] worst rhs " << worst_rhs << " (limit 1e-13); kron pressure residual "
                  << worst_kres << " (1e-11), kron repeat " << worst_krep << " (1e-13); step vs MUMPS "
                  << worst_step << " (gross-error limit " << step_limit << "; MUMPS noise seen " << mumps_noise << "): " << (ok ? "PASS" : "FAIL") << std::endl;
        return ok ? 0 : 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    sim.run();
    double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    lap(t_lap);

    std::ofstream fout(outfile);
    fout << out_nx << " " << out_ny << "\n";
    for (int i = 0; i < out_nx; ++i) {
        for (int j = 0; j < out_ny; ++j) {
            double xbar = static_cast<double>(i) / (out_nx - 1);
            double ybar = static_cast<double>(j) / (out_ny - 1);
            auto [u, v] = sim.sample_velocity(xbar * Lx, ybar * Ly);
            fout << xbar << " " << ybar << " " << u << " " << v << "\n";
        }
    }
    fout.close();
    const double tm_write_out = lap(t_lap);

    auto mb = sim.compute_mass_balance(0.0);
    const double tm_mass_balance = lap(t_lap);
    std::cout << "[coupling_driver] mass balance: int_S=" << mb.int_S << " int_div=" << mb.int_div
              << " flux=" << mb.flux << " |v|max=" << mb.v_max << " |v|L2=" << mb.v_L2
              << " warm=" << (sim.was_warm_started() ? 1 : 0) << " wall_run_s=" << wall_s
              << std::endl;
    if (opt.has("diag")) {
        std::ofstream dfile(opt.str("diag", ""), std::ios::app);
        dfile.precision(10);
        dfile << cycle << " " << mb.int_S << " " << mb.int_div << " " << mb.flux << " "
              << mb.v_max << " " << mb.v_L2 << " " << (sim.was_warm_started() ? 1 : 0) << " "
              << wall_s << " " << mb.int_S_x << " " << mb.int_S_y;
        if (opt.has("rot-diag")) {
            auto rd = sim.compute_rotation_diag();
            dfile << " " << rd.omega_L2 << " " << rd.div_L2 << " " << rd.S_L2 << " " << rd.divres_L2
                  << " " << (rd.S_L2 > 0 ? rd.divres_L2 / rd.S_L2 : 0.0) << " " << rd.vrot_L2 << " "
                  << rd.vmb_L2 << " " << (rd.div_L2 > 0 ? rd.omega_L2 / rd.div_L2 : 0.0) << " "
                  << (rd.vmb_L2 > 0 ? rd.vrot_L2 / rd.vmb_L2 : 0.0);
        }
        dfile << "\n";
    }
    const double tm_diag = lap(t_lap);
    if (opt.has("save-prefix")) {
        sim.save_final(opt.str("save-prefix", ""), fields_nx, fields_ny);
    }
    const double tm_save_fields = lap(t_lap);
    if (!state_file.empty()) {
        if (!sim.save_state(state_file)) {
            std::cerr << "coupling_driver: could not write state file " << state_file << "\n";
        }
    }
    const double tm_save_state = lap(t_lap);
    std::cout << "[coupling_driver] wrote velocity field to " << outfile << std::endl;
    if (opt.has("timing")) {
        const auto& tm = sim.timing;
        const auto& mt = sim.mumps_timers();
        const double main_total = std::chrono::duration<double>(clk::now() - t_main0).count();
        const char* hdr =
            "cycle,nsteps,Nx,Ny,warm,read_input,setup,sink_norm,farfield,run,before,load_state,"
            "steps,assemble,rhs,mumps_analyze,mumps_factorize,mumps_solve,mumps_calls,after_step,"
            "write_out,mass_balance,diag,save_fields,save_state,main_total,kron_solve,rhs_v1,rhs_bc,rhs_v2,rhs_p,shift\n";
        bool need_header = !std::ifstream("ns_timing.csv").good();
        std::ofstream tf("ns_timing.csv", std::ios::app);
        if (need_header) {
            tf << hdr;
        }
        tf.precision(6);
        tf << cycle << "," << nsteps << "," << Nx << "," << Ny << ","
           << (sim.was_warm_started() ? 1 : 0) << "," << tm_read_input << "," << tm_setup << ","
           << tm_sink_norm << "," << tm_farfield << "," << wall_s << "," << tm.before << ","
           << tm.load_state << "," << tm.steps << "," << tm.assemble << "," << tm.rhs << ","
           << mt.analyze_s << "," << mt.factorize_s << "," << mt.solve_s << "," << mt.calls << ","
           << tm.after_step << "," << tm_write_out << "," << tm_mass_balance << "," << tm_diag
           << "," << tm_save_fields << "," << tm_save_state << "," << main_total << ","
           << tm.kron << "," << tm.rhs_v1 << "," << tm.rhs_bc << "," << tm.rhs_v2 << ","
           << tm.rhs_p << "," << tm.shift << "\n";
    }
    return 0;
}
