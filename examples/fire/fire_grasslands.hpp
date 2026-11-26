// SPDX-FileCopyrightText: 2015 - 2024 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT
// Modified for Grasslands National Park fire simulation

#ifndef FIRE_FIRE_GRASSLANDS_HPP
#define FIRE_FIRE_GRASSLANDS_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <utility>
#include <vector>

#include "ads/executor/galois.hpp"
#include "../wildfires_ns/ns_inproc.hpp"

// for debugging
inline bool is_valid(double x) {
    return std::isfinite(x) && !std::isnan(x);
}
#include "ads/bspline/eval.hpp"
#include "ads/lin/tensor/tensor.hpp"
#include "ads/output_manager.hpp"
#include "ads/simulation.hpp"
#include "ads/util.hpp"

inline double falloff(double r, double R, double t) {
    if (t < r)
        return 1.0;
    if (t > R)
        return 0.0;
    double h = (t - r) / (R - r);
    return std::pow((h - 1) * (h + 1), 2);
}

inline double bump(double r, double R, double x, double y, double cx, double cy, double domain_size) {
    double dx = x - cx;
    double dy = y - cy;
    double t = std::sqrt(dx * dx + dy * dy) / domain_size;
    return falloff(r / domain_size, R / domain_size, t);
}

namespace ads::problems {

class fire_grasslands : public simulation_2d {
private:
    using Base = simulation_2d;
    vector_type u, u_prev;

    vector_type fuel, fuel_prev;

    // fuel map from external data
    using map_array = ads::lin::tensor<double, 2>;
    map_array fuel_map_data;
    bool use_fuel_map = false;
    bool ignition_line = false;
    int map_nx = 0;
    int map_ny = 0;

    // domain dimensions (can be rectangular)
    double domain_x = 100.0;
    double domain_y = 100.0;

    // background wind b, in m/s
    double bx_mps = 0.0;
    double by_mps = 0.0;

    // physical parameters
    double ch = 1.0;
    double Ar = 5.7e-5;
    double rho = 1.293;
    double cp = 1.0;
    double cw = 0.5;
    double kappa = 0.3;
    double xi = 2e-2;
    double sigma = 5.67e-8;
    double eps = 0.05;
    double delta_x = 3.5e-2 / eps;
    double delta_z = 1.5 * eps;
    double T0 = 300;
    double Tcomb = 1200;
    double Tig = 800;
    double Ta = 300;

    double M = 2;
    double M1 = 1;

    // ignition point
    double ignition_x = 50.0;
    double ignition_y = 50.0;
    double ignition_r = 5.0;
    double ignition_R = 15.0;

public:
    struct coupling_config {

        int step = -1;
        std::string driver_path = "./coupling_driver";

        double kb = 0.001563;
        double h = 0.002;
        int N = 32;
        int nsteps = 50;
        double Re = 20000.0;
        int period = 0;
        int max_cycles = -1;
        bool recenter = false;
        double search_radius_m = 150.0;


        std::string source = "rate";

        std::string domain = "full";
        double window_L_m = 1000.0;
        double elem_m = 450.0;
        double margin_m = 500.0;
        double min_window_m = 1000.0;
        double sample_spacing_m = 50.0;

        std::string time_mode = "sync";
        double t_end = 2.0;
        bool warm = true;

        std::string bc = "farfield";
        int fields_n = 200;

        bool normalize_sink = true;
        bool save_fields = true;
        bool keep_io = true;

        bool timing = false;

        bool factor_cache = false;

        bool kron = true;
        int diag_every = 1;

        bool inproc = false;
        int ns_threads = 8;

        double dt_max_s = 0.0;

        int csv_every = 1;

        bool rot_diag = false;

        std::string sink_law = "legacy";
        double kb_phys = -1.0;
    };

private:
    coupling_config cfg;
    int coupling_cycles_done = 0;
    int last_coupling_step = 0;

    std::chrono::steady_clock::time_point t_prev_coupling_end{};

    double window_cx_sim = 50.0, window_cy_sim = 50.0;


    double scale_x_m = 217.7;
    double scale_y_m = 310.8;
    static constexpr double COUPLING_L_REF_M = 1000.0;
    static constexpr double COUPLING_U_MPS = 1000.0;

    double fire_time_unit_s = 1.0;

    static constexpr double FUEL_CONSUMPTION_CF = 3e2;


    std::string cw_law = "const";

    double cw_a = 0.0232, cw_b = 0.00772;
    int out_every = 100;

    std::vector<double> back_u, back_beta;

    bool back_fuelgrad = false;

    bool qw_supg = false;
    double mesh_hx = 1.0, mesh_hy = 1.0;  // fire mesh size in sim-units (set from main)


    bool wind_field_active = false;
    int wf_nx = 0, wf_ny = 0;
    double wf_xmin_sim = 0, wf_xmax_sim = 0, wf_ymin_sim = 0, wf_ymax_sim = 0;
    std::vector<double> wf_bx, wf_by;

    struct ns_window {
        double x0_m = 0, y0_m = 0, wx_m = 0, wy_m = 0;  // real meters
        int Nx = 32, Ny = 32;                           // NS elements
        int nsx = 21, nsy = 21;                         // fire->NS sample points
    };

    galois_executor executor;
    int fire_threads = 1;
    std::unique_ptr<ads::ns_inproc> ns_ip;  // declared after `executor`: destroyed before it
    output_manager<2> output;

public:
    explicit fire_grasslands(const config_2d& config, int threads,
                             const std::string& fuel_map_file = "")
    : Base{config}
    , u{shape()}
    , u_prev{shape()}
    , fuel{shape()}
    , fuel_prev{shape()}
    , fuel_map_data{{1, 1}}
    , executor{threads}
    , fire_threads{threads}
    , output{x.B, y.B, 300} {
        if (!fuel_map_file.empty()) {
            load_fuel_map(fuel_map_file);
        }
    }

    void set_ignition_point(double ix, double iy, double ir = 5.0, double iR = 15.0) {
        ignition_x = ix;
        ignition_y = iy;
        ignition_r = ir;
        ignition_R = iR;
    }


    void set_wind(double wx_mps, double wy_mps) {
        bx_mps = wx_mps;
        by_mps = wy_mps;
    }

    void set_scales(double sx_m, double sy_m) {
        scale_x_m = sx_m;
        scale_y_m = sy_m;
    }

    void set_fire_time_unit(double seconds) { fire_time_unit_s = seconds; }


    void set_ignition_line(bool on) { ignition_line = on; }
    void set_cw(double value) { cw = value; }
    void set_cw_linear(double a, double b) { cw_a = a; cw_b = b; }
    void set_out_every(int n) { out_every = n > 0 ? n : 100; }

    void set_back_fuelgrad(bool on) { back_fuelgrad = on; }


    void set_back_table(const std::string& spec) {
        back_u.clear();
        back_beta.clear();
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            auto c = item.find(':');
            if (c == std::string::npos) continue;
            back_u.push_back(std::atof(item.substr(0, c).c_str()));
            back_beta.push_back(std::atof(item.substr(c + 1).c_str()));
        }
    }

    double back_beta_at(double sp) const {
        if (sp <= back_u.front()) return back_beta.front();
        if (sp >= back_u.back()) return back_beta.back();
        std::size_t k = 1;
        while (back_u[k] < sp) ++k;
        double t = (sp - back_u[k - 1]) / (back_u[k] - back_u[k - 1]);
        return back_beta[k - 1] * (1 - t) + back_beta[k] * t;
    }

    void set_wind_transport_options(const std::string& law, bool supg, double hx, double hy) {
        cw_law = law;
        qw_supg = supg;
        mesh_hx = hx;
        mesh_hy = hy;
    }

    void set_coupling(const coupling_config& c) { cfg = c; }

    void set_domain(double dx, double dy) {
        domain_x = dx;
        domain_y = dy;
    }

    double init_state(double x, double y) {

        double avg_domain = (domain_x + domain_y) / 2.0;
        if (ignition_line) {
            return T0 + Tcomb * falloff(ignition_r, ignition_R, std::abs(x - ignition_x));
        }
        return T0 + Tcomb * bump(ignition_r, ignition_R, x, y, ignition_x, ignition_y, avg_domain);
    };

private:
    void load_fuel_map(const std::string& filename) {
        std::ifstream input(filename);
        if (!input.is_open()) {
            std::cerr << "Warning: Could not open fuel map file: " << filename << std::endl;
            std::cerr << "Using uniform fuel distribution." << std::endl;
            return;
        }

        input >> map_nx >> map_ny;
        fuel_map_data = map_array{{map_nx, map_ny}};

        for (int i = 0; i < map_nx; i++) {
            for (int j = 0; j < map_ny; j++) {
                double fx, fy, val;
                input >> fx >> fy >> val;
                fuel_map_data(i, j) = val;
            }
        }

        use_fuel_map = true;
        std::cout << "Loaded fuel map: " << map_nx << " x " << map_ny << std::endl;
    }

    double get_fuel_value(double x, double y) {
        if (!use_fuel_map) {
            return 1.0;  // Default uniform fuel
        }

        // map coordinates to grid indices
        double fx = (x / domain_x) * (map_nx - 1);
        double fy = (y / domain_y) * (map_ny - 1);

        int ix = static_cast<int>(fx);
        int iy = static_cast<int>(fy);

        // clamp to valid range
        ix = std::max(0, std::min(ix, map_nx - 2));
        iy = std::max(0, std::min(iy, map_ny - 2));

        // bilinear interpolation
        double dx = fx - ix;
        double dy = fy - iy;

        double v00 = fuel_map_data(ix, iy);
        double v10 = fuel_map_data(ix + 1, iy);
        double v01 = fuel_map_data(ix, iy + 1);
        double v11 = fuel_map_data(ix + 1, iy + 1);

        double v0 = v00 * (1 - dx) + v10 * dx;
        double v1 = v01 * (1 - dx) + v11 * dx;

        return v0 * (1 - dy) + v1 * dy;
    }

    void before() override {
        prepare_matrices();


        window_cx_sim = ignition_x;
        window_cy_sim = ignition_y;
        t_prev_coupling_end = std::chrono::steady_clock::now();
        if (cfg.step > 0 && cfg.timing) {
            std::ofstream tcsv("coupling_timing.csv");
            tcsv << "cycle,step,steps_since_prev,fire_since_prev_s,sample_write_s,driver_system_s,"
                    "read_out_s,coupling_total_s\n";
        }
        if (cfg.step > 0) {
            std::ofstream csv("coupling_cycles.csv");
            csv << "cycle,step,fire_time,x0_m,y0_m,wx_m,wy_m,NS_Nx,NS_Ny,sample_nx,sample_ny,"
                   "hot_samples,sum_rate_gated,sum_rate_T_only,ns_t_end,ns_wall_s,"
                   "max_speed_mps\n";
        }

        auto init = [this](double x, double y) { return init_state(x, y); };
        projection(u, init);
        solve(u);


        double u_min = std::numeric_limits<double>::max();
        double u_max = std::numeric_limits<double>::lowest();
        for (int ix = 0; ix < static_cast<int>(u.size()); ++ix) {
            double val = u.data()[ix];
            if (is_valid(val)) {
                u_min = std::min(u_min, val);
                u_max = std::max(u_max, val);
            }
        }
        std::cout << "[DEBUG] Initial T range: [" << u_min << ", " << u_max << "]" << std::endl;

        output.to_file(u, "out_0.data");

        // initialize fuel from map
        auto fuel_init = [this](double x, double y) { return get_fuel_value(x, y); };
        projection(fuel, fuel_init);
        solve(fuel);

        double f_min = std::numeric_limits<double>::max();
        double f_max = std::numeric_limits<double>::lowest();
        for (int ix = 0; ix < static_cast<int>(fuel.size()); ++ix) {
            double val = fuel.data()[ix];
            if (is_valid(val)) {
                f_min = std::min(f_min, val);
                f_max = std::max(f_max, val);
            }
        }
        std::cout << "[DEBUG] Initial Fuel range: [" << f_min << ", " << f_max << "]" << std::endl;

        output.to_file(fuel, "fuel_0.data");
    }

    void before_step(int /*iter*/, double /*t*/) override {
        using std::swap;
        swap(u, u_prev);
        swap(fuel, fuel_prev);
    }

    struct reaction_t {
        double delta, r;
    };
    reaction_t reaction(double T, double F) const {
        double u_val = std::max(100.0, T);
        double fuel_val = std::max(0.0, std::min(1.0, F));
        double delta = u_val > Tig && fuel_val > 0.2 ? 1.0 : 0.0;
        double exp_arg = -Ta / u_val;
        double exp_val = (exp_arg > -100.0) ? std::exp(exp_arg) : 0.0;
        return {delta, delta * Ar * u_val * exp_val};
    }


    double sample_rate(double x_sim, double y_sim, double* r_T_only = nullptr, double* F_out = nullptr) {
        bspline::eval_ctx cx(x.B.degree), cy(y.B.degree);
        double T = bspline::eval(x_sim, y_sim, u, x.B, y.B, cx, cy);
        double F = bspline::eval(x_sim, y_sim, fuel, x.B, y.B, cx, cy);
        if (F_out) {
            *F_out = std::max(0.0, std::min(1.0, F));
        }
        if (r_T_only) {
            *r_T_only = reaction(T, 1.0).r;
        }
        return reaction(T, F).r;
    }

    double sample_temperature(double x_sim, double y_sim) {

        bspline::eval_ctx cx(x.B.degree), cy(y.B.degree);
        return bspline::eval(x_sim, y_sim, u, x.B, y.B, cx, cy);
    }


    std::pair<double, double> wind_background_sim() const {
        return {bx_mps * fire_time_unit_s / scale_x_m, by_mps * fire_time_unit_s / scale_y_m};
    }


    std::pair<double, double> wind_at(double x_sim, double y_sim) const {
        if (!wind_field_active) {
            return wind_background_sim();
        }
        if (x_sim < wf_xmin_sim || x_sim > wf_xmax_sim || y_sim < wf_ymin_sim
            || y_sim > wf_ymax_sim) {
            return wind_background_sim();
        }
        double fx = (x_sim - wf_xmin_sim) / (wf_xmax_sim - wf_xmin_sim) * (wf_nx - 1);
        double fy = (y_sim - wf_ymin_sim) / (wf_ymax_sim - wf_ymin_sim) * (wf_ny - 1);
        int ix = std::max(0, std::min(wf_nx - 2, static_cast<int>(fx)));
        int iy = std::max(0, std::min(wf_ny - 2, static_cast<int>(fy)));
        double tx = fx - ix, ty = fy - iy;
        auto idx = [this](int i, int j) { return i * wf_ny + j; };
        auto blerp = [&](const std::vector<double>& g) {
            double v00 = g[idx(ix, iy)], v10 = g[idx(ix + 1, iy)];
            double v01 = g[idx(ix, iy + 1)], v11 = g[idx(ix + 1, iy + 1)];
            return v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty
                 + v11 * tx * ty;
        };
        return {blerp(wf_bx), blerp(wf_by)};
    }


    void recenter_window() {
        double rx_sim = cfg.search_radius_m / scale_x_m;
        double ry_sim = cfg.search_radius_m / scale_y_m;
        double xmin = std::max(0.0, window_cx_sim - rx_sim);
        double xmax = std::min(domain_x, window_cx_sim + rx_sim);
        double ymin = std::max(0.0, window_cy_sim - ry_sim);
        double ymax = std::min(domain_y, window_cy_sim + ry_sim);

        const int ns = 21;
        double best_T = std::numeric_limits<double>::lowest();
        double best_x = window_cx_sim, best_y = window_cy_sim;
        for (int i = 0; i < ns; ++i) {
            double x_sim = xmin + (xmax - xmin) * i / (ns - 1);
            for (int j = 0; j < ns; ++j) {
                double y_sim = ymin + (ymax - ymin) * j / (ns - 1);
                double T = sample_temperature(x_sim, y_sim);
                if (is_valid(T) && T > best_T) {
                    best_T = T;
                    best_x = x_sim;
                    best_y = y_sim;
                }
            }
        }
        std::cout << "[COUPLING] recenter: (" << window_cx_sim << "," << window_cy_sim
                  << ") -> (" << best_x << "," << best_y << ") sim-units, T=" << best_T << "K"
                  << " (search box x=[" << xmin << "," << xmax << "] y=[" << ymin << "," << ymax
                  << "])" << std::endl;
        window_cx_sim = best_x;
        window_cy_sim = best_y;
    }


    ns_window choose_window() {
        ns_window w;
        const double dom_x_m = domain_x * scale_x_m;
        const double dom_y_m = domain_y * scale_y_m;
        if (cfg.domain == "full") {
            w.x0_m = 0;
            w.y0_m = 0;
            w.wx_m = dom_x_m;
            w.wy_m = dom_y_m;
        } else if (cfg.domain == "adaptive") {

            double xmin = dom_x_m, xmax = 0, ymin = dom_y_m, ymax = 0;
            bool any = false;
            const double sp = cfg.sample_spacing_m;
            int nx = static_cast<int>(std::ceil(dom_x_m / sp));
            int ny = static_cast<int>(std::ceil(dom_y_m / sp));
            for (int i = 0; i <= nx; ++i) {
                double xm = std::min(dom_x_m, i * sp);
                for (int j = 0; j <= ny; ++j) {
                    double ym = std::min(dom_y_m, j * sp);
                    double T = sample_temperature(xm / scale_x_m, ym / scale_y_m);
                    if (is_valid(T) && T > Tig) {
                        any = true;
                        xmin = std::min(xmin, xm);
                        xmax = std::max(xmax, xm);
                        ymin = std::min(ymin, ym);
                        ymax = std::max(ymax, ym);
                    }
                }
            }
            if (!any) {
                xmin = xmax = window_cx_sim * scale_x_m;
                ymin = ymax = window_cy_sim * scale_y_m;
            }
            double x0 = xmin - cfg.margin_m, x1 = xmax + cfg.margin_m;
            double y0 = ymin - cfg.margin_m, y1 = ymax + cfg.margin_m;
            if (x1 - x0 < cfg.min_window_m) {
                double c = 0.5 * (x0 + x1);
                x0 = c - cfg.min_window_m / 2;
                x1 = c + cfg.min_window_m / 2;
            }
            if (y1 - y0 < cfg.min_window_m) {
                double c = 0.5 * (y0 + y1);
                y0 = c - cfg.min_window_m / 2;
                y1 = c + cfg.min_window_m / 2;
            }
            w.x0_m = std::max(0.0, x0);
            w.y0_m = std::max(0.0, y0);
            w.wx_m = std::min(dom_x_m, x1) - w.x0_m;
            w.wy_m = std::min(dom_y_m, y1) - w.y0_m;
        } else {
            double cx_real = window_cx_sim * scale_x_m;
            double cy_real = window_cy_sim * scale_y_m;
            w.wx_m = w.wy_m = cfg.window_L_m;
            w.x0_m = cx_real - cfg.window_L_m / 2.0;
            w.y0_m = cy_real - cfg.window_L_m / 2.0;
        }
        w.nsx = static_cast<int>(std::lround(w.wx_m / cfg.sample_spacing_m)) + 1;
        w.nsy = static_cast<int>(std::lround(w.wy_m / cfg.sample_spacing_m)) + 1;
        if (cfg.domain == "window") {
            w.Nx = w.Ny = cfg.N;
        } else {
            w.Nx = std::max(8, static_cast<int>(std::lround(w.wx_m / cfg.elem_m)));
            w.Ny = std::max(8, static_cast<int>(std::lround(w.wy_m / cfg.elem_m)));
        }
        return w;
    }


    void do_coupling(int step_now) {
        using clk = std::chrono::steady_clock;
        const auto tc0 = clk::now();  // v6 (1a) timing
        const double fire_since_prev_s = std::chrono::duration<double>(tc0 - t_prev_coupling_end).count();
        const int steps_since_prev = step_now - last_coupling_step;
        if (cfg.domain == "window" && cfg.recenter) {
            recenter_window();
        }
        const ns_window w = choose_window();
        const int k = coupling_cycles_done;
        auto timing_row = [&](double sample_write_s, double system_s, double read_out_s) {
            if (!cfg.timing) {
                return;
            }
            const auto tc1 = clk::now();
            std::ofstream tcsv("coupling_timing.csv", std::ios::app);
            tcsv << k << "," << step_now << "," << steps_since_prev << "," << fire_since_prev_s << ","
                 << sample_write_s << "," << system_s << "," << read_out_s << ","
                 << std::chrono::duration<double>(tc1 - tc0).count() << "\n";
            t_prev_coupling_end = clk::now();
        };
        char tag[32];
        std::snprintf(tag, sizeof(tag), "%03d", k);


        const std::string in_file =
            std::string(cfg.source == "rate" ? "coupling_rate_in_" : "coupling_temp_in_")
            + std::to_string(k) + ".txt";
        const std::string out_file = "coupling_wind_out_" + std::to_string(k) + ".txt";

        const bool rate_source = cfg.source == "rate";

        const bool law_fuel = cfg.sink_law == "fuel";
        double kb_ns = cfg.kb;
        if (cfg.sink_law != "legacy") {
            const double kbp = cfg.kb_phys > 0 ? cfg.kb_phys
                                               : cfg.kb * fire_time_unit_s / (law_fuel ? FUEL_CONSUMPTION_CF : 1.0);
            kb_ns = kbp / fire_time_unit_s;
        }
        long hot_samples = 0;
        double sum_gated = 0, sum_T_only = 0;
        const bool inproc = cfg.inproc;
        ads::ns_inproc::field_grid ip_grid;
        if (inproc) {
            ip_grid.nx = w.nsx;
            ip_grid.ny = w.nsy;
            ip_grid.T.assign(static_cast<std::size_t>(w.nsx) * w.nsy, 0.0);
        }
        {
            std::ofstream fout;
            if (!inproc) {
                fout.open(in_file);
                fout << w.nsx << " " << w.nsy << "\n";
            }
            for (int i = 0; i < w.nsx; ++i) {
                for (int j = 0; j < w.nsy; ++j) {
                    double xbar = static_cast<double>(i) / (w.nsx - 1);
                    double ybar = static_cast<double>(j) / (w.nsy - 1);
                    double x_sim = (w.x0_m + xbar * w.wx_m) / scale_x_m;
                    double y_sim = (w.y0_m + ybar * w.wy_m) / scale_y_m;
                    double T = sample_temperature(x_sim, y_sim);
                    double r_T_only = 0.0;
                    double F_s = 1.0;
                    double rate = sample_rate(x_sim, y_sim, &r_T_only, law_fuel ? &F_s : nullptr);
                    if (law_fuel) {
                        rate *= FUEL_CONSUMPTION_CF * F_s;
                    }
                    if (is_valid(T) && T > Tig) {
                        ++hot_samples;
                    }
                    sum_gated += rate;
                    sum_T_only += r_T_only;
                    if (inproc) {
                        ip_grid.T[i * w.nsy + j] = rate_source ? rate : T;
                    } else {
                        fout << xbar << " " << ybar << " " << (rate_source ? rate : T) << "\n";
                    }
                }
            }
        }
        const double sample_write_s = std::chrono::duration<double>(clk::now() - tc0).count();


        if (cfg.source == "rate" && sum_gated <= 0.0) {
            std::cout << "[COUPLING] cycle " << k << ": no burning cells, wind = background b (NS skipped)"
                      << std::endl;
            wind_field_active = false;
            last_coupling_step = step_now;
            if (inproc) {
                if (ns_ip) {
                    ns_ip->forget_state();
                }
            } else {
                std::remove("ns_state.bin");
            }
            if (k % cfg.csv_every == 0) {
                std::ofstream csv("coupling_cycles.csv", std::ios::app);
                csv << k << "," << step_now << "," << step_now * steps.dt << "," << w.x0_m << ","
                    << w.y0_m << "," << w.wx_m << "," << w.wy_m << "," << w.Nx << "," << w.Ny << ","
                    << w.nsx << "," << w.nsy << "," << hot_samples << ",0," << sum_T_only
                    << ",0,0,0\n";
            }
            if (!cfg.keep_io && !inproc) {
                std::remove(in_file.c_str());
            }
            timing_row(sample_write_s, 0.0, 0.0);
            return;
        }


        double ns_t_end = cfg.t_end;
        if (cfg.time_mode == "sync") {
            ns_t_end = (step_now - last_coupling_step) * steps.dt * fire_time_unit_s;  // T = 1 s
        }

        int ns_nsteps = cfg.nsteps;
        if (cfg.dt_max_s > 0.0 && cfg.time_mode == "sync") {
            ns_nsteps = std::max(1, static_cast<int>(std::ceil(ns_t_end / cfg.dt_max_s * (1.0 - 1e-12))));
        }

        const double nd_x = w.wx_m / COUPLING_L_REF_M;
        const double nd_y = w.wy_m / COUPLING_L_REF_M;
        const int fn = cfg.fields_n;
        const int fnx = nd_x >= nd_y ? fn : std::max(20, static_cast<int>(std::lround(fn * nd_x / nd_y)));
        const int fny = nd_y >= nd_x ? fn : std::max(20, static_cast<int>(std::lround(fn * nd_y / nd_x)));

        int mx = 0, my = 0;
        std::vector<double> ubar_v, vbar_v;
        double ns_wall_s = 0;
        auto t_read0 = clk::now();
        if (inproc) {
            if (!ns_ip) {
                ns_ip = std::make_unique<ads::ns_inproc>(executor, fire_threads, cfg.ns_threads);
            }
            ads::ns_inproc::params prm;
            prm.Nx = w.Nx;
            prm.Ny = w.Ny;
            prm.Lx = nd_x;
            prm.Ly = nd_y;
            prm.nsteps = ns_nsteps;
            prm.t_end = ns_t_end;
            prm.kb = kb_ns;
            prm.h = cfg.h;
            prm.Re = cfg.Re;
            prm.wind_x_mps = bx_mps;
            prm.wind_y_mps = by_mps;
            prm.U_mps = COUPLING_U_MPS;
            prm.normalize_sink = cfg.normalize_sink;
            prm.farfield = cfg.bc == "farfield";
            prm.warm = cfg.warm;
            prm.factor_cache = cfg.factor_cache;
            prm.kron = cfg.kron;
            prm.diag_every = cfg.diag_every;
            prm.out_nx = w.nsx;
            prm.out_ny = w.nsy;
            prm.fields_nx = fnx;
            prm.fields_ny = fny;
            if (cfg.save_fields) {
                prm.save_prefix = std::string("ns_cycle_") + tag;
            }
            prm.diag_file = "ns_diag.txt";
            prm.cycle = k;
            prm.timing = cfg.timing;
            prm.rot_diag = cfg.rot_diag;
            std::cout << "[COUPLING] cycle " << k << ": in-process wind solve, N=" << w.Nx << "x"
                      << w.Ny << " nsteps=" << ns_nsteps << " t_end=" << ns_t_end << std::endl;
            auto t0 = clk::now();
            auto res = ns_ip->solve(ip_grid, prm);
            ns_wall_s = std::chrono::duration<double>(clk::now() - t0).count();
            mx = w.nsx;
            my = w.nsy;
            t_read0 = clk::now();
            for (int i = 0; i < mx; ++i) {
                for (int j = 0; j < my; ++j) {
                    if (!std::isfinite(res.u[i * my + j]) || !std::isfinite(res.v[i * my + j])) {
                        std::cerr << "[COUPLING] cycle " << k << ": wind solver returned a non-finite"
                                  << " value at (" << i << "," << j << ") (in-process); ABORTING"
                                  << " (wind solver diverged)." << std::endl;
                        std::exit(3);
                    }
                }
            }
            ubar_v = std::move(res.u);
            vbar_v = std::move(res.v);
        } else {
        std::ostringstream cmd;
        cmd << cfg.driver_path << " " << in_file << " " << out_file << " " << w.Nx << " "
            << ns_nsteps << " " << kb_ns << " " << cfg.h << " " << cfg.Re;
        cmd << " --Ny=" << w.Ny << " --domain-x=" << nd_x << " --domain-y=" << nd_y
            << " --t-end=" << ns_t_end << " --U=" << COUPLING_U_MPS << " --wind-x=" << bx_mps
            << " --wind-y=" << by_mps << " --out-nx=" << w.nsx << " --out-ny=" << w.nsy
            << " --input=" << cfg.source << " --bc=" << cfg.bc << " --cycle=" << k << " --diag=ns_diag.txt";
        if (cfg.normalize_sink) {
            cmd << " --normalize-sink=1";
        }
        if (cfg.warm) {
            cmd << " --warm --state=ns_state.bin";
        }
        if (cfg.save_fields) {
            cmd << " --save-prefix=ns_cycle_" << tag << " --fields-nx=" << fnx
                << " --fields-ny=" << fny;
        }
        if (cfg.timing) {
            cmd << " --timing";
        }
        if (cfg.factor_cache) {
            cmd << " --factor-cache";
        }
        if (cfg.rot_diag) {
            cmd << " --rot-diag";
        }
        cmd << " --solver=" << (cfg.kron ? "kron" : "mumps");
        if (cfg.diag_every > 1) {
            cmd << " --diag-every=" << cfg.diag_every;
        }
        std::cout << "[COUPLING] running: " << cmd.str() << std::endl;
        auto t0 = std::chrono::steady_clock::now();
        int rc = std::system(cmd.str().c_str());
        ns_wall_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (rc != 0) {
            std::cerr << "[COUPLING] coupling_driver failed (rc=" << rc
                      << "); leaving wind field unchanged." << std::endl;
            return;
        }

        t_read0 = clk::now();
        std::ifstream fin(out_file);
        if (!fin) {
            std::cerr << "[COUPLING] could not read " << out_file
                      << "; leaving wind field unchanged." << std::endl;
            return;
        }
        fin >> mx >> my;
        ubar_v.assign(static_cast<std::size_t>(mx) * my, 0.0);
        vbar_v.assign(static_cast<std::size_t>(mx) * my, 0.0);
        for (int i = 0; i < mx; ++i) {
            for (int j = 0; j < my; ++j) {
                double xbar, ybar, ubar, vbar;
                fin >> xbar >> ybar >> ubar >> vbar;
                if (!fin || !std::isfinite(ubar) || !std::isfinite(vbar)) {
                    std::cerr << "[COUPLING] cycle " << k << ": wind solver returned a non-finite"
                              << " or unreadable value at (" << i << "," << j << ") in " << out_file
                              << "; ABORTING (wind solver diverged)." << std::endl;
                    std::exit(3);
                }
                ubar_v[i * my + j] = ubar;
                vbar_v[i * my + j] = vbar;
            }
        }
        fin.close();
        }
        wf_nx = mx;
        wf_ny = my;
        wf_bx.assign(static_cast<std::size_t>(mx) * my, 0.0);
        wf_by.assign(static_cast<std::size_t>(mx) * my, 0.0);
        wf_xmin_sim = w.x0_m / scale_x_m;
        wf_xmax_sim = (w.x0_m + w.wx_m) / scale_x_m;
        wf_ymin_sim = w.y0_m / scale_y_m;
        wf_ymax_sim = (w.y0_m + w.wy_m) / scale_y_m;

        double max_speed_mps = 0.0;
        for (int i = 0; i < mx; ++i) {
            for (int j = 0; j < my; ++j) {
                double ubar = ubar_v[i * my + j];
                double vbar = vbar_v[i * my + j];
                double u_real_mps = ubar * COUPLING_U_MPS;
                double v_real_mps = vbar * COUPLING_U_MPS;
                max_speed_mps =
                    std::max(max_speed_mps, std::sqrt(u_real_mps * u_real_mps
                                                        + v_real_mps * v_real_mps));
                wf_bx[i * my + j] = u_real_mps * fire_time_unit_s / scale_x_m;
                wf_by[i * my + j] = v_real_mps * fire_time_unit_s / scale_y_m;
            }
        }
        wind_field_active = true;
        last_coupling_step = step_now;
        std::cout << "[COUPLING] installed spatially-varying wind field, window sim-x=["
                  << wf_xmin_sim << "," << wf_xmax_sim << "] sim-y=[" << wf_ymin_sim << ","
                  << wf_ymax_sim << "], max speed = " << max_speed_mps << " m/s, NS wall "
                  << ns_wall_s << " s" << std::endl;
        {
            if (k % cfg.csv_every == 0) {
                std::ofstream csv("coupling_cycles.csv", std::ios::app);
                csv << k << "," << step_now << "," << step_now * steps.dt << "," << w.x0_m << ","
                    << w.y0_m << "," << w.wx_m << "," << w.wy_m << "," << w.Nx << "," << w.Ny << ","
                    << w.nsx << "," << w.nsy << "," << hot_samples << "," << sum_gated << ","
                    << sum_T_only << "," << ns_t_end << "," << ns_wall_s << "," << max_speed_mps
                    << "\n";
            }
        }
        if (!cfg.keep_io && !inproc) {
            std::remove(in_file.c_str());
            std::remove(out_file.c_str());
        }
        timing_row(sample_write_s, ns_wall_s, std::chrono::duration<double>(clk::now() - t_read0).count());
    }

    void step(int /*iter*/, double t) override {
        compute_rhs(t);
        solve(u);
        solve(fuel);
    }

    void compute_rhs(double t) {
        auto& rhs = u;
        auto& rhs_fuel = fuel;

        zero(rhs);
        zero(rhs_fuel);
        executor.for_each(elements(), [&](index_type e) {
            auto U = element_rhs();
            auto F = element_rhs();

            double J = jacobian(e);
            for (auto q : quad_points()) {
                double w = weight(q);
                auto x = point(e, q);

                value_type u = eval_fun(u_prev, e, q);
                value_type fuel = eval_fun(fuel_prev, e, q);

                for (auto a : dofs_on_element(e)) {
                    auto aa = dof_global_to_local(e, a);
                    value_type v = eval_basis(e, q, a);
                    double inv = 1.0 / (rho * cp);

                    double u_val = std::max(100.0, u.val);
                    double fuel_val = std::max(0.0, std::min(1.0, fuel.val));

                    auto rx = reaction(u.val, fuel.val);
                    double delta = rx.delta;
                    double hc = -70;
                    double r = rx.r;
                    double Rc = -1e4 * rho * ch * hc * M / M1 * r;

                    auto [bx_here, by_here] = wind_at(x[0], x[1]);
                    double cw_here = cw;
                    if (cw_law == "grasso") {
                        double sp = std::hypot(bx_here * scale_x_m / fire_time_unit_s,
                                               by_here * scale_y_m / fire_time_unit_s);  // m/s
                        cw_here = 1e-5 * sp + 1.8e-5 * sp * sp;
                    } else if (cw_law == "linear") {
                        double sp = std::hypot(bx_here * scale_x_m / fire_time_unit_s,
                                               by_here * scale_y_m / fire_time_unit_s);  // m/s
                        cw_here = cw_a + cw_b * sp;
                    }
                    double Qw = -rho * cw_here * (bx_here * u.dx + by_here * u.dy);
                    if (!back_u.empty()) {

                        double gF = std::hypot(fuel.dx, fuel.dy);
                        bool backing_zone = back_fuelgrad
                            && ((gF > 1e-2 && (bx_here * fuel.dx + by_here * fuel.dy) < 0)
                                || (gF <= 1e-2 && delta > 0));
                        if (Qw < 0 || backing_zone) {
                            double sp = std::hypot(bx_here * scale_x_m / fire_time_unit_s,
                                                   by_here * scale_y_m / fire_time_unit_s);  // m/s
                            Qw *= back_beta_at(sp);
                        }
                    }
                    double qc = -kappa * grad_dot(u, v);
                    double qd = 0.0;
                    double qr = -4 * sigma * eps * delta_x * std::pow(u_val, 3) * grad_dot(u, v);
                    double Qconv = xi * (T0 - u_val);
                    double Qrz = sigma * eps / delta_z * (std::pow(T0, 4) - std::pow(u_val, 4));

                    double val = (Rc + Qw + Qconv + Qrz) * v.val + qc + qd + qr;
                    if (qw_supg) {
                        double bm = std::hypot(bx_here, by_here);
                        if (bm > 1e-12) {
                            double hb = bm / (std::abs(bx_here) / mesh_hx + std::abs(by_here) / mesh_hy);
                            double tau = hb / (2 * bm);
                            val += Qw * tau * (bx_here * v.dx + by_here * v.dy);
                        }
                    }
                    U(aa[0], aa[1]) += (u_val * v.val + steps.dt * inv * val) * w * J;

                    double fval = -delta * 3e2 * r * fuel_val * v.val;
                    F(aa[0], aa[1]) += (fuel_val * v.val + steps.dt * fval) * w * J;
                }
            }
            executor.synchronized([&] { update_global_rhs(rhs, U, e); });
            executor.synchronized([&] { update_global_rhs(rhs_fuel, F, e); });
        });
    }

    void after_step(int iter, double /*t*/) override {
        auto i = iter + 1;


        double u_min = std::numeric_limits<double>::max();
        double u_max = std::numeric_limits<double>::lowest();
        bool has_nan = false;

        for (int ix = 0; ix < static_cast<int>(u.size()); ++ix) {
            double val = u.data()[ix];
            if (!is_valid(val)) {
                has_nan = true;
            } else {
                u_min = std::min(u_min, val);
                u_max = std::max(u_max, val);
            }
        }

        if (has_nan) {
            std::cerr << "!!! Step " << i << ": NaN/Inf detected!" << std::endl;
            std::cerr << "    T range: [" << u_min << ", " << u_max << "]" << std::endl;
            std::cerr << "    ABORTING" << std::endl;
            std::exit(1);
        }

        if (i % out_every == 0) {
            std::cout << "Step " << i << " | T: [" << u_min << ", " << u_max << "]" << std::endl;
            output.to_file(u, "out_%d.data", i);
            output.to_file(fuel, "fuel_%d.data", i);
        }


        if (cfg.step > 0 && (cfg.max_cycles <= 0 || coupling_cycles_done < cfg.max_cycles)) {
            bool trigger = coupling_cycles_done == 0
                ? (i == cfg.step)
                : (cfg.period > 0
                   && i == cfg.step + coupling_cycles_done * cfg.period);
            if (trigger) {
                do_coupling(i);
                ++coupling_cycles_done;
            }
        }
    }
};

}

#endif
