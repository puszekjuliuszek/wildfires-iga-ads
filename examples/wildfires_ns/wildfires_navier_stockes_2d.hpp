// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef WILDFIRES_NS_WILDFIRES_NAVIER_STOKES_2D_HPP
#define WILDFIRES_NS_WILDFIRES_NAVIER_STOKES_2D_HPP

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <galois/Timer.h>

#include "ads/executor/galois.hpp"
#include "ads/output_manager.hpp"
#include "ads/simulation.hpp"
#include "ads/simulation/utils.hpp"
#include "ads/solver/mumps.hpp"
#include "kron_ops.hpp"
#include "space_set.hpp"

namespace ads {

using value_type = function_value_2d;
using point_type = std::array<double, 2>;
using value_pair = std::array<value_type, 2>;

struct prob_manufactured_poly {
    double Re;

    bool navier_stokes = false;

    value_type exact_p(point_type p, double t) const {
        auto x = p[0];
        auto et = std::exp(-t);
        return {et * (x * (1 - x) - 1. / 6), et * (1 - 2 * x), 0.0};
    }

    value_pair exact_v(point_type p, double t) const {
        auto et = std::exp(-t);

        auto f = [](double x, double y) {
            return x * x * (1 - x) * (1 - x) * (2 * y - 6 * y * y + 4 * y * y * y);
        };

        auto dfx = [](double x, double y) {
            return (4 * x * x * x - 6 * x * x + 2 * x) * (2 * y - 6 * y * y + 4 * y * y * y);
        };

        auto dfy = [](double x, double y) {
            return x * x * (1 - x) * (1 - x) * (2 - 12 * y + 12 * y * y);
        };

        const auto [x, y] = p;
        value_type vx = {f(x, y), dfx(x, y), dfy(x, y)};
        value_type vy = {-f(y, x), -dfy(y, x), -dfx(y, x)};

        return {et * vx, et * vy};
    }

    point_type forcing(point_type p, double t) const {
        const auto [x, y] = p;
        auto v = exact_v(p, t);
        auto et = std::exp(-t);

        auto fx = (12 - 24 * y) * x * x * x * x
                + (-24 + 48 * y) * x * x * x
                + (-48 * y + 72 * y * y - 48 * y * y * y + 12) * x * x
                + (-2 + 24 * y - 72 * y * y + 48 * y * y * y) * x
                + 1 - 4 * y + 12 * y * y - 8 * y * y * y;

        auto fy = (8 - 48 * y + 48 * y * y) * x * x * x
                + (-12 + 72 * y - 72 * y * y) * x * x
                + (4 - 24 * y + 48 * y * y - 48 * y * y * y + 24 * y * y * y * y) * x
                - 12 * y * y + 24 * y * y * y - 12 * y * y * y * y;

        return {et * fx - v[0].val, et * fy - v[1].val};
    }

    double sink(point_type, double) const { return 0.0; }
};

struct prob_manufactured_nonpoly {
    double Re;

    bool navier_stokes = false;

    value_type exact_p(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        return {cos(x) * sin(y + t), -sin(x) * sin(y + t), cos(x) * cos(y + t)};
    }

    value_pair exact_v(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        value_type vx = {sin(x) * sin(y + t), cos(x) * sin(y + t), sin(x) * cos(y + t)};
        value_type vy = {cos(x) * cos(y + t), -sin(x) * cos(y + t), -cos(x) * sin(y + t)};

        return {vx, vy};
    }

    point_type forcing(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        auto fx = sin(x) * cos(y + t) + 2 / Re * sin(x) * sin(y + t) - sin(x) * sin(y + t);
        auto fy = -cos(x) * sin(y + t) + 2 / Re * cos(x) * cos(y + t) + cos(x) * cos(y + t);

        return {fx, fy};
    }

    double sink(point_type, double) const { return 0.0; }
};

struct prob_manufactured_NS_nonpoly {
    double Re;

    bool navier_stokes = true;

    value_type exact_p(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        return {cos(x) * sin(y + t), -sin(x) * sin(y + t), cos(x) * cos(y + t)};
    }

    value_pair exact_v(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        value_type vx = {sin(x) * sin(y + t), cos(x) * sin(y + t), sin(x) * cos(y + t)};
        value_type vy = {cos(x) * cos(y + t), -sin(x) * cos(y + t), -cos(x) * sin(y + t)};

        return {vx, vy};
    }

    point_type forcing(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        auto fx = sin(x) * cos(y + t) + 2 / Re * sin(x) * sin(y + t) - sin(x) * sin(y + t)
                + sin(x) * cos(x);

        auto fy = -cos(x) * sin(y + t) + 2 / Re * cos(x) * cos(y + t) + cos(x) * cos(y + t)
                - sin(y + t) * cos(y + t);

        return {fx, fy};
    }

    double sink(point_type, double) const { return 0.0; }
};

struct prob_manufactured_NS_nonpoly_varying_Re {
    double Re = 1.0;
    bool navier_stokes = true;

    double reynolds(double t) const {
        const auto s = t / 2;  // normalized to [0, 1] range
        const auto m = 1.0;
        const auto M = 2000.0;
        return (1 - s) * m + s * M;
    }

    value_type exact_p(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        return {cos(x) * sin(y + t), -sin(x) * sin(y + t), cos(x) * cos(y + t)};
    }

    value_pair exact_v(point_type p, double t) const {
        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        value_type vx = {sin(x) * sin(y + t), cos(x) * sin(y + t), sin(x) * cos(y + t)};
        value_type vy = {cos(x) * cos(y + t), -sin(x) * cos(y + t), -cos(x) * sin(y + t)};

        return {vx, vy};
    }

    point_type forcing(point_type p, double t) {

        Re = reynolds(t);

        const auto [x, y] = p;
        using std::cos;
        using std::sin;

        auto fx = sin(x) * cos(y + t) + 2 / Re * sin(x) * sin(y + t) - sin(x) * sin(y + t)
                + sin(x) * cos(x);

        auto fy = -cos(x) * sin(y + t) + 2 / Re * cos(x) * cos(y + t) + cos(x) * cos(y + t)
                - sin(y + t) * cos(y + t);

        return {fx, fy};
    }

    double sink(point_type, double) const { return 0.0; }
};

struct prob_cavity_flow {
    double Re;
    bool navier_stokes;

    value_type exact_p(point_type, double) const { return {}; }

    value_pair exact_v(point_type p, double) const {
        auto y = p[1];
        auto vy = value_type{};
        auto vx = value_type{};

        if (y == 1) {
            vx.val = 1;
        }
        return {vx, vy};
    }

    point_type forcing(point_type, double) const { return {0, 0}; }

    double sink(point_type, double) const { return 0.0; }
};

struct prob_cavity_flow_accelerating {
    double Re;
    bool navier_stokes;

    value_type exact_p(point_type, double) const { return {}; }

    value_pair exact_v(point_type p, double t) const {
        auto y = p[1];
        auto vy = value_type{};
        auto vx = value_type{};

        if (y == 1) {
            vx.val = t;
        }
        return {vx, vy};
    }

    point_type forcing(point_type, double) const { return {0, 0}; }

    double sink(point_type, double) const { return 0.0; }
};

struct prob_pyrogenic_plume {
    double Re;
    bool navier_stokes = false;

    double Ar = 5.7e-5;
    double Ta = 300;
    double Tig = 800;

    double patch_cx = 0.5, patch_cy = 0.5;
    double patch_r = 0.05, patch_R = 0.15;
    double Tcomb = 1200, T0 = 300;

    std::function<double(point_type)> temperature_field;

    bool input_is_rate = false;

    double force_alpha = 0.0;
    bool sink_enabled = true;
    std::function<point_type(point_type)> temperature_gradient;

    double kb = 1.0;
    double h = 1.0;

    double sink_scale = 1.0;

    double wind_x = 0.0, wind_y = 0.0;

    bool farfield_bc = false;
    double ff_cx = 0.0, ff_cy = 0.0, ff_Q = 0.0;

    static double falloff(double r, double R, double t) {
        if (t < r) return 1.0;
        if (t > R) return 0.0;
        double h = (t - r) / (R - r);
        return std::pow((h - 1) * (h + 1), 2);
    }

    double patch_temperature(point_type p) const {
        if (temperature_field) {
            return temperature_field(p);
        }
        auto dx = p[0] - patch_cx;
        auto dy = p[1] - patch_cy;
        auto t = std::sqrt(dx * dx + dy * dy);
        return T0 + Tcomb * falloff(patch_r, patch_R, t);
    }

    double reaction_rate(double T) const {
        if (T <= Tig) {
            return 0.0;
        }
        return Ar * T * std::exp(-Ta / T);
    }

    value_type exact_p(point_type, double) const { return {}; }
    static constexpr bool zero_exact_pressure = true;
    static constexpr bool static_sink = true;

    value_pair initial_v(point_type) const {
        value_type vx{}, vy{};
        vx.val = wind_x;
        vy.val = wind_y;
        return {vx, vy};
    }

    value_pair exact_v(point_type p, double) const {
        value_type vx{}, vy{};
        vx.val = wind_x;
        vy.val = wind_y;
        if (farfield_bc) {
            double dx = p[0] - ff_cx;
            double dy = p[1] - ff_cy;
            double r2 = std::max(dx * dx + dy * dy, 1e-12);
            double c = -ff_Q / (2 * M_PI * r2);
            vx.val += c * dx;
            vy.val += c * dy;
        }
        return {vx, vy};
    }

    point_type forcing(point_type p, double) const {
        if (force_alpha != 0.0 && temperature_gradient) {
            auto g = temperature_gradient(p);
            return {-force_alpha * g[0], -force_alpha * g[1]};
        }
        return {0, 0};
    }

    double sink(point_type p, double) const {
        if (!sink_enabled) {
            return 0.0;
        }
        auto val = patch_temperature(p);
        auto r = input_is_rate ? val : reaction_rate(val);
        return sink_scale * kb * r / h;
    }
};

namespace detail {

struct scoped_acc {
    double& acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit scoped_acc(double& a) : acc{a} { }
    ~scoped_acc() {
        acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

template <typename P, typename = void>
struct has_initial_v : std::false_type { };

template <typename P, typename = void>
struct has_static_sink : std::false_type { };

template <typename P>
struct has_static_sink<P, std::void_t<decltype(P::static_sink)>> : std::bool_constant<P::static_sink> { };

template <typename P, typename = void>
struct has_zero_exact_pressure : std::false_type { };

template <typename P>
struct has_zero_exact_pressure<P, std::void_t<decltype(P::zero_exact_pressure)>>
: std::bool_constant<P::zero_exact_pressure> { };

template <typename P>
struct has_initial_v<P, std::void_t<decltype(std::declval<const P&>().initial_v(point_type{}))>>
: std::true_type { };

}

template <typename Problem>
class wildfires_navier_stokes_2d : public simulation_2d {
private:
    using Base = simulation_2d;
    using value_pair = std::array<value_type, 2>;

    Problem problem;



    std::unique_ptr<galois_executor> owned_executor;
    const galois_executor& executor;

    space_set trial, test;

    vector_type vx, vy, p;
    vector_type p_star, phi;
    vector_type vx_prev, vy_prev;

    mumps::solver solver;
    output_manager<2> outputU1, outputU2, outputP;

    galois::Timer solver_timer;
    galois::Timer total_timer;


    bool save_every_step = true;
    std::string warm_start_file;
    bool warm_started = false;

    bool reuse_mode = false, matrices_ready = false, state_in_memory = false, have_state = false;





    bool use_kron = false;
    bool use_fast = false;
    bool par_rhs = false, par_solve = false;
    std::vector<double> wpx, wpy;
    std::vector<double> sink_cache;
    int diag_every = 1;
    long global_step = 0;
    bool verify_ready = false;
    struct kron_cache {
        bool vel_ready = false, p_ready = false;
        double dt_key = 0, Re_key = 0;
        std::unique_ptr<kron::dim_op> vel[2][2][2];
        std::unique_ptr<kron::dim_op> pres[2];
    } kc;

    bool factor_cache = false;
    struct cached_factor {
        std::unique_ptr<mumps::solver> solver;
        std::unique_ptr<mumps::problem> problem;
        double dt_key = 0;
    };
    std::unique_ptr<mumps::solver> rot_solver;
    std::unique_ptr<mumps::problem> rot_problem;
    mumps::phase_timers retired_;
    cached_factor cache_[5];


    template <typename Assemble>
    void solve_cached(int slot, bool dt_dependent, double* rhs, int n, Assemble&& assemble) {
        auto& c = cache_[slot];
        const double dt = steps.dt;
        if (!c.solver || (dt_dependent && c.dt_key != dt)) {
            if (c.solver) {
                const auto& x = c.solver->timers();
                retired_.analyze_s += x.analyze_s;
                retired_.factorize_s += x.factorize_s;
                retired_.solve_s += x.solve_s;
                retired_.calls += x.calls;
            }
            c.problem.reset();
            c.solver.reset();
            c.solver = std::make_unique<mumps::solver>();
            c.problem = std::make_unique<mumps::problem>(rhs, n);
            {
                detail::scoped_acc t_asm{timing.assemble};
                assemble(*c.problem);
            }
            c.solver->factorize(*c.problem);
            c.dt_key = dt;
        }
        c.solver->solve_factored(rhs);
    }

public:

    struct timing_t {
        double before = 0;
        double load_state = 0;
        double steps = 0;
        double assemble = 0;
        double rhs = 0;
        double after_step = 0;
        double kron = 0;
        double rhs_v1 = 0, rhs_bc = 0, rhs_v2 = 0, rhs_p = 0, shift = 0;
    };
    timing_t timing;

    void set_diag_every(int n) { diag_every = n < 1 ? 1 : n; }

    void set_kron_solver(bool on) {
        if (on && !test_equals_trial()) {
            throw std::runtime_error("kron solver requires test space == trial space");
        }
        use_kron = on;
        use_fast = on && fast_ok();
        set_parallel_thresholds();
    }

    void set_parallel_thresholds() {
        par_rhs = trial.Px.elements * trial.Py.elements >= 2500;
        par_solve = trial.U1x.dofs() * trial.U1y.dofs() >= 20000;
    }

    void set_factor_cache(bool on) { factor_cache = on; }

    void run_cycle(int nsteps, double dt, bool warm) {
        reuse_mode = true;
        state_in_memory = warm && have_state;
        steps = timesteps_config{nsteps, dt};
        total_timer = galois::Timer{};
        total_timer.start();
        run();
        have_state = true;
    }
    void discard_state() { have_state = false; }
    mumps::phase_timers mumps_timers() const {
        mumps::phase_timers t = solver.timers();
        t.analyze_s += retired_.analyze_s;
        t.factorize_s += retired_.factorize_s;
        t.solve_s += retired_.solve_s;
        t.calls += retired_.calls;
        for (const auto& c : cache_) {
            if (c.solver) {
                const auto& x = c.solver->timers();
                t.analyze_s += x.analyze_s;
                t.factorize_s += x.factorize_s;
                t.solve_s += x.solve_s;
                t.calls += x.calls;
            }
        }
        return t;
    }

    wildfires_navier_stokes_2d(const space_set& trial_, const space_set& test_,
                      const timesteps_config& steps, Problem problem, int out_nx = 200,
                      int out_ny = 200, galois_executor* external_executor = nullptr)
    : Base{test_.Px, test_.Py, steps}
    , problem{problem}
    , owned_executor{external_executor ? nullptr : std::make_unique<galois_executor>(8)}
    , executor{external_executor ? *external_executor : *owned_executor}
    , trial{trial_}
    , test{test_}
    , vx{{trial.U1x.dofs(), trial.U1y.dofs()}}
    , vy{{trial.U2x.dofs(), trial.U2y.dofs()}}
    , p{{trial.Px.dofs(), trial.Py.dofs()}}
    , p_star{{trial.Px.dofs(), trial.Py.dofs()}}
    , phi{{trial.Px.dofs(), trial.Py.dofs()}}
    , vx_prev{{trial.U1x.dofs(), trial.U1y.dofs()}}
    , vy_prev{{trial.U2x.dofs(), trial.U2y.dofs()}}
    , outputU1{trial.U1x.B, trial.U1y.B, static_cast<std::size_t>(out_nx),
               static_cast<std::size_t>(out_ny)}
    , outputU2{trial.U2x.B, trial.U2y.B, static_cast<std::size_t>(out_nx),
               static_cast<std::size_t>(out_ny)}
    , outputP{trial.Px.B, trial.Py.B, static_cast<std::size_t>(out_nx),
              static_cast<std::size_t>(out_ny)} {
        total_timer.start();
    }

    void set_save_every_step(bool on) { save_every_step = on; }

    void set_warm_start(const std::string& file) { warm_start_file = file; }
    bool was_warm_started() const { return warm_started; }

    Problem& get_problem() { return problem; }

    std::pair<double, double> sample_velocity(double xx, double yy) {
        bspline::eval_ctx cx1(trial.U1x.B.degree), cy1(trial.U1y.B.degree);
        bspline::eval_ctx cx2(trial.U2x.B.degree), cy2(trial.U2y.B.degree);
        double u = bspline::eval(xx, yy, vx, trial.U1x.B, trial.U1y.B, cx1, cy1);
        double v = bspline::eval(xx, yy, vy, trial.U2x.B, trial.U2y.B, cx2, cy2);
        return {u, v};
    }

    void pressure_sum(vector_type& target, const vector_type& a, const vector_type& b) const {
        for (auto i : dofs(trial.Px, trial.Py)) {
            target(i[0], i[1]) = a(i[0], i[1]) + b(i[0], i[1]);
        }
    }

    void compute_pressure_predictor() { pressure_sum(p_star, p, phi); }

    void apply_pressure_corrector() {
        vector_type rhs{{trial.Px.dofs(), trial.Py.dofs()}};

        double chi = 0;
        {
            detail::scoped_acc t_rhs{timing.rhs};
            compute_rhs_pressure_update(rhs, chi);
        }
        if (factor_cache) {
            solve_cached(4, false, rhs.data(), static_cast<int>(rhs.size()), [&](mumps::problem& pr) {
                assemble_matrix(pr, 0, 0, false, false, trial.Px, trial.Py);
            });
        } else {
        mumps::problem problem(rhs.data(), rhs.size());
        {
            detail::scoped_acc t_asm{timing.assemble};
            assemble_matrix(problem, 0, 0, false, false, trial.Px, trial.Py);
        }
        solver.solve(problem);
        }

        p = rhs;
    }

    void before() override {
        detail::scoped_acc t_before{timing.before};
        sink_cache.clear();
        if (!(reuse_mode && matrices_ready)) {
            trial.U1x.factorize_matrix();
            trial.U1y.factorize_matrix();
            trial.U2x.factorize_matrix();
            trial.U2y.factorize_matrix();
            trial.Px.factorize_matrix();
            trial.Py.factorize_matrix();
            matrices_ready = true;
        }

        auto project = [&](auto& rhs, auto& x, auto& y, auto fun) {
            vector_type buffer{{x.dofs(), y.dofs()}};
            compute_projection(rhs, x.basis, y.basis, [&](double x, double y) {
                return fun({x, y});
            });
            ads_solve(rhs, buffer, x.data(), y.data());
        };

        warm_started = false;
        if (reuse_mode) {
            warm_started = state_in_memory;
        } else if (!warm_start_file.empty()) {
            detail::scoped_acc t_load{timing.load_state};
            warm_started = load_state(warm_start_file);
        }

        if (!warm_started) {
            if (reuse_mode) {
                zero(vx_prev);
                zero(vy_prev);
            }
            project(vx, trial.U1x, trial.U1y,
                    [this](point_type x) { return initial_velocity(x)[0].val; });
            project(vy, trial.U2x, trial.U2y,
                    [this](point_type x) { return initial_velocity(x)[1].val; });

            project(p, trial.Px, trial.Py,
                    [this](point_type x) { return problem.exact_p(x, 0).val; });
            project(phi, trial.Px, trial.Py, [this](point_type x) {
                double t = 0.5 * steps.dt;
                return problem.exact_p(x, t).val - problem.exact_p(x, 0).val;
            });
        }

        if (save_every_step) {
            save_to_file(0);
        }
    }

    value_pair initial_velocity(point_type x) const {
        if constexpr (detail::has_initial_v<Problem>::value) {
            return problem.initial_v(x);
        } else {
            return problem.exact_v(x, 0);
        }
    }

    bool save_state(const std::string& file) const {
        std::ofstream os{file, std::ios::binary};
        if (!os) {
            return false;
        }
        auto put = [&](const vector_type& v) {
            std::int64_t n = static_cast<std::int64_t>(v.size());
            os.write(reinterpret_cast<const char*>(&n), sizeof(n));
            os.write(reinterpret_cast<const char*>(v.data()), n * sizeof(double));
        };
        put(vx);
        put(vy);
        put(p);
        put(phi);
        put(vx_prev);
        put(vy_prev);
        return static_cast<bool>(os);
    }

    bool load_state(const std::string& file) {
        std::ifstream is{file, std::ios::binary};
        if (!is) {
            std::cout << "[wildfires_ns] warm start: cannot open " << file << ", cold start"
                      << std::endl;
            return false;
        }
        vector_type* targets[] = {&vx, &vy, &p, &phi, &vx_prev, &vy_prev};
        std::vector<std::vector<double>> buf;
        for (auto* t : targets) {
            std::int64_t n = 0;
            is.read(reinterpret_cast<char*>(&n), sizeof(n));
            if (!is || n != static_cast<std::int64_t>(t->size())) {
                std::cout << "[wildfires_ns] warm start: state in " << file
                          << " does not match the discretization, cold start" << std::endl;
                return false;
            }
            buf.emplace_back(static_cast<std::size_t>(n));
            is.read(reinterpret_cast<char*>(buf.back().data()), n * sizeof(double));
            if (!is) {
                std::cout << "[wildfires_ns] warm start: truncated " << file << ", cold start"
                          << std::endl;
                return false;
            }
        }
        for (std::size_t k = 0; k < buf.size(); ++k) {
            std::copy(buf[k].begin(), buf[k].end(), targets[k]->data());
        }
        std::cout << "[wildfires_ns] warm start from " << file << std::endl;
        return true;
    }

    struct mass_balance {
        double int_S = 0;
        double int_div = 0;
        double flux = 0;
        double v_max = 0;
        double v_L2 = 0;
        double int_S_x = 0, int_S_y = 0;
    };

    mass_balance compute_mass_balance(double t = 0) {
        mass_balance mb;
        double L2sq = 0;
        for (auto e : elements(trial.Px, trial.Py)) {
            double J = jacobian(e, trial.Px, trial.Py);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double w = weight(q, trial.Px, trial.Py);
                auto x = point(e, q, trial.Px, trial.Py);
                value_type vvx = eval(vx, e, q, trial.U1x, trial.U1y);
                value_type vvy = eval(vy, e, q, trial.U2x, trial.U2y);
                double s = problem.sink(x, t);
                mb.int_S += s * w * J;
                mb.int_S_x += s * x[0] * w * J;
                mb.int_S_y += s * x[1] * w * J;
                mb.int_div += (vvx.dx + vvy.dy) * w * J;
                L2sq += (vvx.val * vvx.val + vvy.val * vvy.val) * w * J;
            }
        }
        mb.v_L2 = std::sqrt(L2sq);

        const double xa = trial.U1x.a, xb = trial.U1x.b, ya = trial.U1y.a, yb = trial.U1y.b;
        const int M = 400;
        auto simpson = [&](auto&& f, double a, double b) {
            int n = 2 * M;
            double hh = (b - a) / n;
            double sum = f(a) + f(b);
            for (int k = 1; k < n; ++k) {
                sum += f(a + k * hh) * (k % 2 ? 4 : 2);
            }
            return sum * hh / 3;
        };
        mb.flux += simpson([&](double y) { return sample_velocity(xb, y).first; }, ya, yb);
        mb.flux -= simpson([&](double y) { return sample_velocity(xa, y).first; }, ya, yb);
        mb.flux += simpson([&](double x) { return sample_velocity(x, yb).second; }, xa, xb);
        mb.flux -= simpson([&](double x) { return sample_velocity(x, ya).second; }, xa, xb);

        const int G = 4 * 64;
        for (int i = 0; i <= G; ++i) {
            for (int j = 0; j <= G; ++j) {
                auto [u, v] = sample_velocity(xa + (xb - xa) * i / G, ya + (yb - ya) * j / G);
                mb.v_max = std::max(mb.v_max, std::hypot(u, v));
            }
        }
        return mb;
    }


    void normalize_sink(int nfx, int nfy) {
        const double xa = trial.Px.a, xb = trial.Px.b, ya = trial.Py.a, yb = trial.Py.b;
        const double dx = (xb - xa) / nfx, dy = (yb - ya) / nfy;
        problem.sink_scale = 1.0;
        double Qf = 0;
        for (int i = 0; i < nfx; ++i) {
            for (int j = 0; j < nfy; ++j) {
                Qf += problem.sink({xa + (i + 0.5) * dx, ya + (j + 0.5) * dy}, 0.0) * dx * dy;
            }
        }
        double Qq = compute_mass_balance(0).int_S;
        if (Qq > 0 && Qf > 0) {
            problem.sink_scale = Qf / Qq;
        }
        std::cout << "[wildfires_ns] sink normalization: fine integral " << Qf
                  << ", element quadrature " << Qq << ", scale " << problem.sink_scale
                  << std::endl;
    }

    void enable_farfield_bc() {
        auto mb = compute_mass_balance(0);
        problem.ff_Q = mb.int_S;
        if (mb.int_S > 0) {
            problem.ff_cx = mb.int_S_x / mb.int_S;
            problem.ff_cy = mb.int_S_y / mb.int_S;
            problem.farfield_bc = true;
        }
        std::cout << "[wildfires_ns] far-field BC: Q=" << problem.ff_Q << " centroid=("
                  << problem.ff_cx << "," << problem.ff_cy << ")" << std::endl;
    }

    void save_final(const std::string& prefix, int out_nx, int out_ny) {
        outputU1.to_file(vx, prefix + "_vx.data");
        outputU2.to_file(vy, prefix + "_vy.data");
        outputP.to_file(p, prefix + "_p.data");

        std::ofstream os_s{prefix + "_S.data"};
        std::ofstream os_d{prefix + "_div.data"};
        os_s.precision(10);
        os_d.precision(10);
        bspline::eval_ders_ctx cx1(trial.U1x.B.degree, 1), cy1(trial.U1y.B.degree, 1);
        bspline::eval_ders_ctx cx2(trial.U2x.B.degree, 1), cy2(trial.U2y.B.degree, 1);
        const double xa = trial.U1x.a, xb = trial.U1x.b, ya = trial.U1y.a, yb = trial.U1y.b;
        for (int i = 0; i <= out_nx; ++i) {
            double x = xa + (xb - xa) * i / out_nx;
            for (int j = 0; j <= out_ny; ++j) {
                double y = ya + (yb - ya) * j / out_ny;
                auto a = bspline::eval_ders(x, y, vx, trial.U1x.B, trial.U1y.B, cx1, cy1);
                auto b = bspline::eval_ders(x, y, vy, trial.U2x.B, trial.U2y.B, cx2, cy2);
                os_s << x << " " << y << " " << problem.sink({x, y}, 0.0) << "\n";
                os_d << x << " " << y << " " << (a.dx + b.dy) << "\n";
            }
        }
    }

    struct rotation_diag {
        double omega_L2 = 0;
        double div_L2 = 0;
        double S_L2 = 0;
        double divres_L2 = 0;
        double vrot_L2 = 0;
        double vmb_L2 = 0;
    };

    rotation_diag compute_rotation_diag() {
        const auto& Ux = trial.U1x;
        const auto& Uy = trial.U1y;
        vector_type rhs{{Ux.dofs(), Uy.dofs()}};

        rotation_diag rd;
        double om2 = 0, div2 = 0, S2 = 0, res2 = 0;
        for (auto e : elements(trial.Px, trial.Py)) {
            double J = jacobian(e, trial.Px, trial.Py);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double w = weight(q, trial.Px, trial.Py);
                auto x = point(e, q, trial.Px, trial.Py);
                value_type vvx = eval(vx, e, q, trial.U1x, trial.U1y);
                value_type vvy = eval(vy, e, q, trial.U2x, trial.U2y);
                double om = vvy.dx - vvx.dy;
                double dv = vvx.dx + vvy.dy;
                double s = problem.sink(x, 0.0);
                om2 += om * om * w * J;
                div2 += dv * dv * w * J;
                S2 += s * s * w * J;
                res2 += (dv + s) * (dv + s) * w * J;
                for (auto a : dofs_on_element(e, Ux, Uy)) {
                    value_type phi = eval_basis(e, q, a, Ux, Uy);
                    rhs(a[0], a[1]) += om * phi.val * w * J;
                }
            }
        }
        rd.omega_L2 = std::sqrt(om2);
        rd.div_L2 = std::sqrt(div2);
        rd.S_L2 = std::sqrt(S2);
        rd.divres_L2 = std::sqrt(res2);

        zero_bc(rhs, trial.U1x, trial.U1y);
        if (!rot_solver) {
            rot_solver = std::make_unique<mumps::solver>();
            rot_problem = std::make_unique<mumps::problem>(rhs.data(), static_cast<int>(rhs.size()));
            assemble_laplace_dirichlet(*rot_problem, Ux, Uy);
            rot_solver->factorize(*rot_problem);
        }
        rot_solver->solve_factored(rhs.data());

        double vrot2 = 0, vmb2 = 0;
        for (auto e : elements(trial.Px, trial.Py)) {
            double J = jacobian(e, trial.Px, trial.Py);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double w = weight(q, trial.Px, trial.Py);
                value_type chi = eval(rhs, e, q, trial.U1x, trial.U1y);
                value_type vvx = eval(vx, e, q, trial.U1x, trial.U1y);
                value_type vvy = eval(vy, e, q, trial.U2x, trial.U2y);
                vrot2 += (chi.dx * chi.dx + chi.dy * chi.dy) * w * J;
                double bx = vvx.val - problem.wind_x, by = vvy.val - problem.wind_y;
                vmb2 += (bx * bx + by * by) * w * J;
            }
        }
        rd.vrot_L2 = std::sqrt(vrot2);
        rd.vmb_L2 = std::sqrt(vmb2);
        return rd;
    }

    void assemble_laplace_dirichlet(mumps::problem& problem, const dimension& Ux,
                                    const dimension& Uy) const {
        for (auto i : dofs(Ux, Uy)) {
            for (auto j : overlapping_dofs(i, Ux, Uy)) {
                int ii = linear_index(i, Ux, Uy) + 1;
                int jj = linear_index(j, Ux, Uy) + 1;
                bool fixed = is_boundary(i[0], Ux) || is_boundary(j[0], Ux) || is_boundary(i[1], Uy)
                          || is_boundary(j[1], Uy);
                if (!fixed) {
                    auto form = [](auto u, auto v) { return u.dx * v.dx + u.dy * v.dy; };
                    problem.add(ii, jj, integrate(i, j, Ux, Uy, Ux, Uy, form));
                }
            }
        }
        for_boundary_dofs(Ux, Uy, [&](index_type dof) {
            problem.add(linear_index(dof, Ux, Uy) + 1, linear_index(dof, Ux, Uy) + 1, 1);
        });
    }

    void output_exact(int i, double t) {
        auto p = [this, t](point_type x) { return problem.exact_p(x, t).val; };
        auto vx = [this, t](point_type x) { return problem.exact_v(x, t)[0].val; };
        auto vy = [this, t](point_type x) { return problem.exact_v(x, t)[1].val; };

        auto project = [&](auto& x, auto& y, auto fun) {
            vector_type rhs{{x.dofs(), y.dofs()}};
            vector_type buffer{{x.dofs(), y.dofs()}};
            compute_projection(rhs, x.basis, y.basis, [&](double x, double y) {
                return fun({x, y});
            });
            ads_solve(rhs, buffer, x.data(), y.data());
            return rhs;
        };

        outputP.to_file(project(trial.Px, trial.Py, p), "pressure_ref_%d.data", i);
        outputU1.to_file(project(trial.U1x, trial.U1y, vx), "vx_ref_%d.data", i);
        outputU2.to_file(project(trial.U2x, trial.U2y, vy), "vy_ref_%d.data", i);
    }

    auto shifted(int n, int k, mumps::problem& problem) const {
        return [&problem, n, k](int i, int j, double val) { problem.add(n + i, k + j, val); };
    }

    bool is_pressure_fixed(index_type /*dof*/) const { return false; }

    void assemble_matrix(mumps::problem& problem, double cx, double cy, bool bcx, bool bcy,
                         const dimension& Ux, const dimension& Uy) const {
        for (auto i : dofs(Ux, Uy)) {
            for (auto j : overlapping_dofs(i, Ux, Uy)) {
                int ii = linear_index(i, Ux, Uy) + 1;
                int jj = linear_index(j, Ux, Uy) + 1;

                bool at_bdx = is_boundary(i[0], Ux) || is_boundary(j[0], Ux);
                bool at_bdy = is_boundary(i[1], Uy) || is_boundary(j[1], Uy);
                bool fixed = (at_bdx && bcx) || (at_bdy && bcy);

                if (!fixed) {
                    auto form = [cx, cy](auto u, auto v) {
                        return u.val * v.val + cx * u.dx * v.dx + cy * u.dy * v.dy;
                    };
                    auto product = integrate(i, j, Ux, Uy, Ux, Uy, form);
                    problem.add(ii, jj, product);
                }
            }
        }
        for_boundary_dofs(Ux, Uy, [&](index_type dof) {
            int i = linear_index(dof, Ux, Uy) + 1;

            bool at_bdx = is_boundary(dof[0], Ux) || is_boundary(dof[0], Ux);
            bool at_bdy = is_boundary(dof[1], Uy) || is_boundary(dof[1], Uy);
            bool fixed = (at_bdx && bcx) || (at_bdy && bcy);
            if (fixed) {
                problem.add(i, i, 1);
            }
        });
    }

    void assemble_matrix_velocity(mumps::problem& problem, double cx, double cy) const {
        auto dU1 = trial.U1x.dofs() * trial.U1y.dofs();

        auto DU1 = test.U1x.dofs() * test.U1y.dofs();
        auto DU2 = test.U2x.dofs() * test.U2y.dofs();

        auto D = DU1 + DU2;

        auto test_vx = shifted(0, 0, problem);
        auto test_vy = shifted(DU1, DU1, problem);

        auto trial_vx = shifted(D, D, problem);
        auto trial_vy = shifted(D + dU1, D + dU1, problem);

        for (auto i : dofs(test.U1x, test.U1y)) {
            for (auto j : overlapping_dofs(i, test.U1x, test.U1y)) {
                int ii = linear_index(i, test.U1x, test.U1y) + 1;
                int jj = linear_index(j, test.U1x, test.U1y) + 1;
                auto eval = [&](auto form) {
                    return integrate(i, j, test.U1x, test.U1y, test.U1x, test.U1y, form);
                };

                if (!is_boundary(i, test.U1x, test.U1y) && !is_boundary(j, test.U1x, test.U1y)) {
                    double val = eval([cx, cy](auto tx, auto vx) {
                        return tx.val * vx.val + cx * tx.dx * vx.dx + cy * tx.dy * vx.dy;
                    });
                    test_vx(ii, jj, val);
                }
            }
        }

        for (auto i : dofs(test.U2x, test.U2y)) {
            for (auto j : overlapping_dofs(i, test.U2x, test.U2y)) {
                int ii = linear_index(i, test.U2x, test.U2y) + 1;
                int jj = linear_index(j, test.U2x, test.U2y) + 1;
                auto eval = [&](auto form) {
                    return integrate(i, j, test.U2x, test.U2y, test.U2x, test.U2y, form);
                };

                if (!is_boundary(i, test.U2x, test.U2y) && !is_boundary(j, test.U2x, test.U2y)) {
                    double val = eval([cx, cy](auto ty, auto vy) {
                        return ty.val * vy.val + cx * ty.dx * vy.dx + cy * ty.dy * vy.dy;
                    });
                    test_vy(ii, jj, val);
                }
            }
        }

        for_boundary_dofs(test.U1x, test.U1y, [&](index_type dof) {
            int i = linear_index(dof, test.U1x, test.U1y) + 1;
            test_vx(i, i, 1);
        });
        for_boundary_dofs(test.U2x, test.U2y, [&](index_type dof) {
            int i = linear_index(dof, test.U2x, test.U2y) + 1;
            test_vy(i, i, 1);
        });

        auto put = [&](int i, int j, int si, int sj, double val, bool fixed_i, bool fixed_j) {
            int ii = i + si;
            int jj = j + sj;

            if (!fixed_i) {
                problem.add(ii, D + jj, val);
            }
            if (!fixed_i && !fixed_j) {
                problem.add(D + jj, ii, val);
            }
        };

        for (auto i : dofs(test.U1x, test.U1y)) {
            for (auto j : overlapping_dofs(i, test.U1x, test.U1y, trial.U1x, trial.U1y)) {
                if (!overlap(i, test.U1x, test.U1y, j, trial.U1x, trial.U1y))
                    continue;

                int ii = linear_index(i, test.U1x, test.U1y) + 1;
                int jj = linear_index(j, trial.U1x, trial.U1y) + 1;
                auto eval = [&](auto form) {
                    return integrate(i, j, test.U1x, test.U1y, trial.U1x, trial.U1y, form);
                };

                bool bd_i = is_boundary(i, test.U1x, test.U1y);
                bool bd_j = is_boundary(j, trial.U1x, trial.U1y);

                double value = eval([cx, cy](auto u, auto v) {
                    return u.val * v.val + cx * u.dx * v.dx + cy * u.dy * v.dy;
                });
                put(ii, jj, 0, 0, value, bd_i, bd_j);
            }
        }

        for (auto i : dofs(test.U2x, test.U2y)) {
            for (auto j : overlapping_dofs(i, test.U1x, test.U1y, trial.U1x, trial.U1y)) {
                if (!overlap(i, test.U2x, test.U2y, j, trial.U2x, trial.U2y))
                    continue;

                int ii = linear_index(i, test.U2x, test.U2y) + 1;
                int jj = linear_index(j, trial.U2x, trial.U2y) + 1;
                auto eval = [&](auto form) {
                    return integrate(i, j, test.U2x, test.U2y, trial.U2x, trial.U2y, form);
                };

                bool bd_i = is_boundary(i, test.U2x, test.U2y);
                bool bd_j = is_boundary(j, trial.U2x, trial.U2y);

                double value = eval([cx, cy](auto u, auto v) {
                    return u.val * v.val + cx * u.dx * v.dx + cy * u.dy * v.dy;
                });
                put(ii, jj, DU1, dU1, value, bd_i, bd_j);
            }
        }

        for_boundary_dofs(trial.U1x, trial.U1y, [&](index_type dof) {
            int i = linear_index(dof, trial.U1x, trial.U1y) + 1;
            trial_vx(i, i, 1);
        });
        for_boundary_dofs(trial.U2x, trial.U2y, [&](index_type dof) {
            int i = linear_index(dof, trial.U2x, trial.U2y) + 1;
            trial_vy(i, i, 1);
        });
    }

    void assemble_matrix_pressure(mumps::problem& problem, double cx, double cy) const {
        auto DP = test.Px.dofs() * test.Py.dofs();

        auto test_p = shifted(0, 0, problem);

        for (auto i : dofs(test.Px, test.Py)) {
            for (auto j : overlapping_dofs(i, test.Px, test.Py)) {
                int ii = linear_index(i, test.Px, test.Py) + 1;
                int jj = linear_index(j, test.Px, test.Py) + 1;

                if (!is_pressure_fixed(i) && !is_pressure_fixed(j)) {
                    auto eval = [&](auto form) {
                        return integrate(i, j, test.Px, test.Py, test.Px, test.Py, form);
                    };
                    double val = eval([](auto w, auto p) { return w.val * p.val; });
                    test_p(ii, jj, val);
                }
            }
        }

        auto put = [&](int i, int j, int si, int sj, double val) {
            int ii = i + si;
            int jj = j + sj;
            problem.add(ii, DP + jj, val);
            problem.add(DP + jj, ii, val);
        };

        for (auto i : dofs(test.Px, test.Py)) {
            for (auto j : overlapping_dofs(i, test.Px, test.Py, trial.Px, trial.Py)) {
                if (!overlap(i, test.Px, test.Py, j, trial.Px, trial.Py))
                    continue;

                int ii = linear_index(i, test.Px, test.Py) + 1;
                int jj = linear_index(j, trial.Px, trial.Py) + 1;
                auto eval = [&](auto form) {
                    return integrate(i, j, test.Px, test.Py, trial.Px, trial.Py, form);
                };

                double value = eval([cx, cy](auto u, auto v) {
                    return u.val * v.val + cx * u.dx * v.dx + cy * u.dy * v.dy;
                });
                put(ii, jj, 0, 0, value);
            }
        }
    }
    void print_solver_info(const std::string& header, const mumps::problem& problem,
                           mumps::solver& solver) {
        auto time = static_cast<double>(solver_timer.get());
        std::cout << "Solver " << header << ": "
                  << " NZ " << problem.nonzero_entries()
                  << " time " << time << " ms"
                  << " assembly FLOPS " << solver.flops_assembly()
                  << " elimination FLOPS " << solver.flops_elimination()
                  << std::endl;
        solver_timer = galois::Timer{};
    }

    template <typename RHS>
    void apply_velocity_bc(RHS& rhs, dimension& Vx, dimension& Vy, double t, int i) {
        dirichlet_bc(rhs, boundary::left, Vx, Vy, [this, t, i, &Vx](auto s) {
            return problem.exact_v({Vx.a, s}, t)[i].val;
        });
        dirichlet_bc(rhs, boundary::right, Vx, Vy, [this, t, i, &Vx](auto s) {
            return problem.exact_v({Vx.b, s}, t)[i].val;
        });
        dirichlet_bc(rhs, boundary::top, Vx, Vy, [this, t, i, &Vy](auto s) {
            return problem.exact_v({s, Vy.b}, t)[i].val;
        });
        dirichlet_bc(rhs, boundary::bottom, Vx, Vy, [this, t, i, &Vy](auto s) {
            return problem.exact_v({s, Vy.a}, t)[i].val;
        });
    }

    void update_velocity_minev(double t) {
        auto dt = steps.dt;
        auto f = [&](point_type x, double s) { return problem.forcing(x, s); };
        auto F = [&](double s) { return [&, s](point_type x) { return f(x, s); }; };
        auto conv = problem.navier_stokes ? dt : 0;

        vector_type rhs_vx{{trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_type rhs_vy{{trial.U2x.dofs(), trial.U2y.dofs()}};

        compute_rhs(rhs_vx, rhs_vy, vx, vy, vx, vy, p, F(t + dt / 2), 0, 0, -dt, -dt, -conv, dt,
                    dt);
        zero_bc(rhs_vx, trial.U1x, trial.U1y);
        zero_bc(rhs_vy, trial.U2x, trial.U2y);

        mumps::problem problem_vx1(rhs_vx.data(), rhs_vx.size());
        assemble_matrix(problem_vx1, 0, 0, true, true, trial.U1x, trial.U1y);
        solver.solve(problem_vx1);

        mumps::problem problem_vy1(rhs_vy.data(), rhs_vy.size());
        assemble_matrix(problem_vy1, 0, 0, true, true, trial.U2x, trial.U2y);
        solver.solve(problem_vy1);

        vector_type rhs_vx2{{trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_type rhs_vy2{{trial.U2x.dofs(), trial.U2y.dofs()}};
        compute_rhs(rhs_vx2, rhs_vy2, vx, vy, rhs_vx, rhs_vy, p, F(t + dt / 2), dt / 2, 0, 0, 0, 0,
                    0, 0);

        zero_bc(rhs_vx2, trial.U1x, trial.U1y);
        zero_bc(rhs_vy2, trial.U2x, trial.U2y);

        mumps::problem problem_vx2(rhs_vx2.data(), rhs_vx2.size());
        assemble_matrix(problem_vx2, dt / 2, 0, true, true, trial.U1x, trial.U1y);
        solver.solve(problem_vx2);

        mumps::problem problem_vy2(rhs_vy2.data(), rhs_vy2.size());
        assemble_matrix(problem_vy2, dt / 2, 0, true, true, trial.U2x, trial.U2y);
        solver.solve(problem_vy2);

        vector_type rhs_vx3{{trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_type rhs_vy3{{trial.U2x.dofs(), trial.U2y.dofs()}};
        compute_rhs(rhs_vx3, rhs_vy3, vx, vy, rhs_vx2, rhs_vy2, p, F(t + dt / 2), 0, dt / 2, 0, 0,
                    0, 0, 0);

        zero_bc(rhs_vx3, trial.U1x, trial.U1y);
        zero_bc(rhs_vy3, trial.U2x, trial.U2y);

        mumps::problem problem_vx3(rhs_vx3.data(), rhs_vx3.size());
        assemble_matrix(problem_vx3, 0, dt / 2, true, true, trial.U1x, trial.U1y);
        solver.solve(problem_vx3);

        mumps::problem problem_vy3(rhs_vy3.data(), rhs_vy3.size());
        assemble_matrix(problem_vy3, 0, dt / 2, true, true, trial.U2x, trial.U2y);
        solver.solve(problem_vy3);

        vx = rhs_vx3;
        vy = rhs_vy3;
    }

    void update_velocity_galerkin(double t) {
        auto dt = steps.dt;
        auto f = [&](point_type x, double s) { return problem.forcing(x, s); };
        auto F = [&](double s) { return [&, s](point_type x) { return f(x, s); }; };
        auto conv = problem.navier_stokes ? dt / 2 : 0;

        vector_type rhs_vx{{trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_type rhs_vy{{trial.U2x.dofs(), trial.U2y.dofs()}};

        compute_rhs(rhs_vx, rhs_vy, vx, vy, vx, vy, p_star, F(t + dt / 2), 0, 0, 0, -dt / 2, -conv,
                    dt / 2, dt / 2);


        apply_velocity_bc(rhs_vx, trial.U1x, trial.U1y, t, 0);


        apply_velocity_bc(rhs_vy, trial.U2x, trial.U2y, t, 1);

        mumps::problem problem_vx1(rhs_vx.data(), rhs_vx.size());
        assemble_matrix(problem_vx1, dt / 2, 0, true, true, trial.U1x, trial.U1y);

        solver.solve(problem_vx1);

        mumps::problem problem_vy1(rhs_vy.data(), rhs_vy.size());
        assemble_matrix(problem_vy1, dt / 2, 0, true, true, trial.U2x, trial.U2y);


        solver.solve(problem_vy1);


        vector_type rhs_vx2{{trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_type rhs_vy2{{trial.U2x.dofs(), trial.U2y.dofs()}};

        compute_rhs(rhs_vx2, rhs_vy2, vx, vy, rhs_vx, rhs_vy, p_star, F(t + dt / 2), 0, 0, -dt / 2,
                    0, -conv, dt / 2, dt / 2);


        apply_velocity_bc(rhs_vx2, trial.U1x, trial.U1y, t, 0);


        apply_velocity_bc(rhs_vy2, trial.U2x, trial.U2y, t, 1);

        mumps::problem problem_vx2(rhs_vx2.data(), rhs_vx2.size());
        assemble_matrix(problem_vx2, 0, dt / 2, true, true, trial.U1x, trial.U1y);

        solver.solve(problem_vx2);

        mumps::problem problem_vy2(rhs_vy2.data(), rhs_vy2.size());
        assemble_matrix(problem_vy2, 0, dt / 2, true, true, trial.U2x, trial.U2y);

        solver.solve(problem_vy2);

        vx_prev = vx;
        vy_prev = vy;
        vx = rhs_vx2;
        vy = rhs_vy2;
    }

    void update_velocity_igrm(int /*i*/, double t) {
        auto dt = steps.dt;
        auto f = [&](point_type x, double s) { return problem.forcing(x, s); };
        auto F = [&](double s) { return [&, s](point_type x) { return f(x, s); }; };
        auto Re = problem.Re;
        auto conv = problem.navier_stokes ? dt / 2 : 0;

        auto dU1 = trial.U1x.dofs() * trial.U1y.dofs();
        auto dU2 = trial.U2x.dofs() * trial.U2y.dofs();
        auto dim_trial = dU1 + dU2;

        auto DU1 = test.U1x.dofs() * test.U1y.dofs();
        auto DU2 = test.U2x.dofs() * test.U2y.dofs();
        auto dim_test = DU1 + DU2;


        std::vector<double> rhs(dim_test + dim_trial);
        vector_view rhs_vx1{rhs.data(), {test.U1x.dofs(), test.U1y.dofs()}};
        vector_view rhs_vy1{rhs.data() + DU1, {test.U2x.dofs(), test.U2y.dofs()}};
        vector_view vx1{rhs.data() + dim_test, {trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_view vy1{vx1.data() + dU1, {trial.U2x.dofs(), trial.U2y.dofs()}};

        {
        detail::scoped_acc t_rhs{timing.rhs};
        compute_rhs(rhs_vx1, rhs_vy1,
                    vx, vy,
                    vx, vy,
                    p_star,
                    F(t + dt / 2),
                    0, 0,
                    0, -dt / (2 * Re),
                    -conv,
                    dt / 2,
                    dt / 2
        );

        apply_velocity_bc(vx1, trial.U1x, trial.U1y, t + dt, 0);
        apply_velocity_bc(vy1, trial.U2x, trial.U2y, t + dt, 1);
        }

        if (factor_cache) {
            solve_cached(0, true, rhs.data(), static_cast<int>(rhs.size()), [&](mumps::problem& pr) {
                assemble_matrix_velocity(pr, dt / (2 * Re), 0);
            });
        } else {
        mumps::problem problem_vx1{rhs};
        {
            detail::scoped_acc t_asm{timing.assemble};
            assemble_matrix_velocity(problem_vx1, dt / (2 * Re), 0);
        }

        solver_timer.start();
        solver.solve(problem_vx1);
        solver_timer.stop();
        print_solver_info("velocity 1", problem_vx1, solver);
        }

        std::vector<double> rhs2(dim_test + dim_trial);
        vector_view rhs_vx2{rhs2.data(), {test.U1x.dofs(), test.U1y.dofs()}};
        vector_view rhs_vy2{rhs2.data() + DU1, {test.U2x.dofs(), test.U2y.dofs()}};
        vector_view vx2{rhs2.data() + dim_test, {trial.U1x.dofs(), trial.U1y.dofs()}};
        vector_view vy2{vx2.data() + dU1, {trial.U2x.dofs(), trial.U2y.dofs()}};

        {
        detail::scoped_acc t_rhs{timing.rhs};
        compute_rhs(rhs_vx2, rhs_vy2,
                    vx, vy,
                    vx1, vy1,
                    p_star,
                    F(t + dt / 2),
                    0, 0,
                    -dt / (2 * Re), 0,
                    -conv,
                    dt / 2,
                    dt / 2
        );

        apply_velocity_bc(vx2, trial.U1x, trial.U1y, t + dt, 0);
        apply_velocity_bc(vy2, trial.U2x, trial.U2y, t + dt, 1);
        }

        if (factor_cache) {
            solve_cached(1, true, rhs2.data(), static_cast<int>(rhs2.size()), [&](mumps::problem& pr) {
                assemble_matrix_velocity(pr, 0, dt / (2 * Re));
            });
        } else {
        mumps::problem problem_vx2{rhs2};
        {
            detail::scoped_acc t_asm{timing.assemble};
            assemble_matrix_velocity(problem_vx2, 0, dt / (2 * Re));
        }

        solver_timer.start();
        solver.solve(problem_vx2);
        solver_timer.stop();
        print_solver_info("velocity 2", problem_vx2, solver);
        }

        vx_prev = vx;
        vy_prev = vy;

        for (auto i : dofs(trial.U1x, trial.U1y)) {
            vx(i[0], i[1]) = vx2(i[0], i[1]);
        }
        for (auto i : dofs(trial.U2x, trial.U2y)) {
            vy(i[0], i[1]) = vy2(i[0], i[1]);
        }
    }

    void update_velocity_exact(double t) {
        double tt = t + steps.dt;

        zero(vx);
        zero(vy);

        auto project = [&](auto& rhs, auto& x, auto& y, auto fun) {
            vector_type buffer{{x.dofs(), y.dofs()}};
            compute_projection(rhs, x.basis, y.basis, [&](double x, double y) {
                return fun({x, y});
            });
            ads_solve(rhs, buffer, x.data(), y.data());
        };
        project(vx, trial.U1x, trial.U1y,
                [this, tt](point_type x) { return problem.exact_v(x, tt)[0].val; });
        project(vy, trial.U2x, trial.U2y,
                [this, tt](point_type x) { return problem.exact_v(x, tt)[1].val; });
    }

    void update_pressure(double t) {
        vector_type rhs_p{{trial.Px.dofs(), trial.Py.dofs()}};

        compute_rhs_pressure_1(rhs_p, vx, vy, trial.Px, trial.Py, steps.dt, t);
        mumps::problem problem_px(rhs_p.data(), rhs_p.size());
        assemble_matrix(problem_px, 1, 0, false, false, trial.Px, trial.Py);
        solver.solve(problem_px);

        zero(phi);
        compute_rhs_pressure_2(phi, rhs_p, trial.Px, trial.Py);
        mumps::problem problem_py(phi.data(), phi.size());
        assemble_matrix(problem_py, 0, 1, false, false, trial.Px, trial.Py);
        solver.solve(problem_py);

        apply_pressure_corrector();
    }

    void update_pressure_exact(double t) {
        double th = t + steps.dt / 2;
        zero(p);
        auto project = [&](auto& rhs, auto& x, auto& y, auto fun) {
            vector_type buffer{{x.dofs(), y.dofs()}};
            compute_projection(rhs, x.basis, y.basis, [&](double x, double y) {
                return fun({x, y});
            });
            ads_solve(rhs, buffer, x.data(), y.data());
        };
        project(p, trial.Px, trial.Py,
                [this, th](point_type x) { return problem.exact_p(x, th).val; });
    }

    void update_pressure_igrm(double t) {
        auto dim_trial = trial.Px.dofs() * trial.Py.dofs();
        auto dim_test = test.Px.dofs() * test.Py.dofs();

        std::vector<double> rhs(dim_test + dim_trial);
        vector_view rhs_p1{rhs.data(), {test.Px.dofs(), test.Py.dofs()}};
        vector_view p1{rhs.data() + dim_test, {trial.Px.dofs(), trial.Py.dofs()}};

        {
            detail::scoped_acc t_rhs{timing.rhs};
            compute_rhs_pressure_1(rhs_p1, vx, vy, test.Px, test.Py, steps.dt, t);
        }
        if (factor_cache) {
            solve_cached(2, false, rhs.data(), static_cast<int>(rhs.size()), [&](mumps::problem& pr) {
                assemble_matrix_pressure(pr, 1, 0);
            });
        } else {
        mumps::problem problem_px{rhs};
        {
            detail::scoped_acc t_asm{timing.assemble};
            assemble_matrix_pressure(problem_px, 1, 0);
        }

        solver_timer.start();
        solver.solve(problem_px);
        solver_timer.stop();
        print_solver_info("pressure 1", problem_px, solver);
        }

        std::vector<double> rhs2(dim_test + dim_trial);
        vector_view rhs_p2{rhs2.data(), {test.Px.dofs(), test.Py.dofs()}};
        vector_view p2{rhs2.data() + dim_test, {trial.Px.dofs(), trial.Py.dofs()}};

        {
            detail::scoped_acc t_rhs{timing.rhs};
            compute_rhs_pressure_2(rhs_p2, p1, test.Px, test.Py);
        }
        if (factor_cache) {
            solve_cached(3, false, rhs2.data(), static_cast<int>(rhs2.size()), [&](mumps::problem& pr) {
                assemble_matrix_pressure(pr, 0, 1);
            });
        } else {
        mumps::problem problem_py{rhs2};
        {
            detail::scoped_acc t_asm{timing.assemble};
            assemble_matrix_pressure(problem_py, 0, 1);
        }

        solver_timer.start();
        solver.solve(problem_py);
        solver_timer.stop();
        print_solver_info("pressure 2", problem_py, solver);
        }

        for (auto i : dofs(trial.Px, trial.Py)) {
            phi(i[0], i[1]) = p2(i[0], i[1]);
        }

        apply_pressure_corrector();
    }

    static bool same_dim(const dimension& a, const dimension& b) {
        return a.p == b.p && a.elements == b.elements && a.dofs() == b.dofs() && a.a == b.a
            && a.b == b.b;
    }

    bool test_equals_trial() const {
        return same_dim(test.U1x, trial.U1x) && same_dim(test.U1y, trial.U1y)
            && same_dim(test.U2x, trial.U2x) && same_dim(test.U2y, trial.U2y)
            && same_dim(test.Px, trial.Px) && same_dim(test.Py, trial.Py);
    }

    void prepare_kron() {
        detail::scoped_acc t_asm{timing.assemble};
        const double dt = steps.dt, Re = problem.Re;
        if (!kc.vel_ready || kc.dt_key != dt || kc.Re_key != Re) {
            const double c = dt / (2 * Re);
            const dimension* dims[2][2] = {{&trial.U1x, &trial.U1y}, {&trial.U2x, &trial.U2y}};
            for (int comp = 0; comp < 2; ++comp) {
                for (int ax = 0; ax < 2; ++ax) {
                    kc.vel[comp][ax][0] = std::make_unique<kron::dim_op>(*dims[comp][ax], 1.0, 0.0, true);
                    kc.vel[comp][ax][1] = std::make_unique<kron::dim_op>(*dims[comp][ax], 1.0, c, true);
                }
            }
            kc.vel_ready = true;
            kc.dt_key = dt;
            kc.Re_key = Re;
        }
        if (!kc.p_ready) {
            kc.pres[0] = std::make_unique<kron::dim_op>(trial.Px, 1.0, 1.0, false);
            kc.pres[1] = std::make_unique<kron::dim_op>(trial.Py, 1.0, 1.0, false);
            kc.p_ready = true;
        }
    }

    bool fast_ok() const {
        const auto& X = trial.U1x.basis;
        const auto& Y = trial.U1y.basis;
        return same_dim(trial.U1x, trial.U2x) && same_dim(trial.U1y, trial.U2y)
            && X.elem_division == 1 && Y.elem_division == 1 && trial.Px.basis.elem_division == 1
            && trial.Py.basis.elem_division == 1 && X.degree <= 3 && Y.degree <= 3
            && trial.Px.basis.degree <= 3 && trial.Py.basis.degree <= 3
            && trial.Px.elements == trial.U1x.elements && trial.Py.elements == trial.U1y.elements
            && trial.Px.basis.quad_order == X.quad_order && trial.Py.basis.quad_order == Y.quad_order
            && X.degree >= 1 && Y.degree >= 1
            && &vx(1, 0) - &vx(0, 0) == 1 && &vx(0, 1) - &vx(0, 0) == trial.U1x.dofs()
            && &vy(1, 0) - &vy(0, 0) == 1 && &vy(0, 1) - &vy(0, 0) == trial.U2x.dofs()
            && &p(1, 0) - &p(0, 0) == 1 && &p(0, 1) - &p(0, 0) == trial.Px.dofs();
    }

    void set_kron_fast(bool on) {
        use_fast = on && fast_ok();
        set_parallel_thresholds();
    }

    template <typename Body>
    void for_element_columns(int ne_x, int degree, Body&& body) const {
        if (!par_rhs) {
            for (int ex = 0; ex < ne_x; ++ex) {
                body(ex);
            }
            return;
        }
        for (int cls = 0; cls <= degree; ++cls) {
            std::vector<int> cols;
            for (int ex = cls; ex < ne_x; ex += degree + 1) {
                cols.push_back(ex);
            }
            executor.for_each(cols, body);
        }
    }

    template <typename Fun>
    void compute_rhs_fast(vector_type& rhsx, vector_type& rhsy, const vector_type& ux,
                          const vector_type& uy, const vector_type& pr, Fun&& forcing, double bx,
                          double by, double conv, double c, double d) const {
        const auto& X = trial.U1x.basis;
        const auto& Y = trial.U1y.basis;
        const auto& PX = trial.Px.basis;
        const auto& PY = trial.Py.basis;
        const int ndx = X.dofs_per_element(), ndy = Y.dofs_per_element();
        const int npx = PX.dofs_per_element(), npy = PY.dofs_per_element();
        const int nqx = X.quad_order, nqy = Y.quad_order;
        const int ney = Y.elements;
        const int sv = trial.U1x.dofs(), sp = trial.Px.dofs();
        const double* UX = &ux(0, 0);
        const double* UY = &uy(0, 0);
        const double* PR = &pr(0, 0);
        double* RX = &rhsx(0, 0);
        double* RY = &rhsy(0, 0);

        for_element_columns(X.elements, X.degree, [&](int ex) {
            for (int ey = 0; ey < ney; ++ey) {
                const int fx = X.first_dof(ex), fy = Y.first_dof(ey);
                const int pfx = PX.first_dof(ex), pfy = PY.first_dof(ey);
                double cxl[16], cyl[16], cpl[16];
                for (int b = 0; b < ndy; ++b) {
                    for (int a = 0; a < ndx; ++a) {
                        cxl[b * 4 + a] = UX[(fx + a) + static_cast<std::size_t>(sv) * (fy + b)];
                        cyl[b * 4 + a] = UY[(fx + a) + static_cast<std::size_t>(sv) * (fy + b)];
                    }
                }
                for (int b = 0; b < npy; ++b) {
                    for (int a = 0; a < npx; ++a) {
                        cpl[b * 4 + a] = PR[(pfx + a) + static_cast<std::size_t>(sp) * (pfy + b)];
                    }
                }
                double lx[16] = {}, ly[16] = {};  // [b][a]
                const double J = X.J[ex] * Y.J[ey];
                for (int qx = 0; qx < nqx; ++qx) {
                    const double* xv = X.b[ex][qx][0];
                    const double* xd = X.b[ex][qx][1];
                    const double* pxv = PX.b[ex][qx][0];
                    double txv[4], txd[4], tyv[4], tyd[4], tpv[4];
                    for (int b = 0; b < ndy; ++b) {
                        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        for (int a = 0; a < ndx; ++a) {
                            s0 += cxl[b * 4 + a] * xv[a];
                            s1 += cxl[b * 4 + a] * xd[a];
                            s2 += cyl[b * 4 + a] * xv[a];
                            s3 += cyl[b * 4 + a] * xd[a];
                        }
                        txv[b] = s0;
                        txd[b] = s1;
                        tyv[b] = s2;
                        tyd[b] = s3;
                    }
                    for (int b = 0; b < npy; ++b) {
                        double s = 0;
                        for (int a = 0; a < npx; ++a) {
                            s += cpl[b * 4 + a] * pxv[a];
                        }
                        tpv[b] = s;
                    }
                    for (int qy = 0; qy < nqy; ++qy) {
                        const double* yv = Y.b[ey][qy][0];
                        const double* yd = Y.b[ey][qy][1];
                        const double* pyv = PY.b[ey][qy][0];
                        const double wj = X.w[qx] * Y.w[qy] * J;

                        double ux_v = 0, ux_x = 0, ux_y = 0, uy_v = 0, uy_x = 0, uy_y = 0, pv = 0;
                        for (int b = 0; b < ndy; ++b) {
                            ux_v += txv[b] * yv[b];
                            ux_x += txd[b] * yv[b];
                            ux_y += txv[b] * yd[b];
                            uy_v += tyv[b] * yv[b];
                            uy_x += tyd[b] * yv[b];
                            uy_y += tyv[b] * yd[b];
                        }
                        for (int b = 0; b < npy; ++b) {
                            pv += tpv[b] * pyv[b];
                        }
                        const auto F = forcing(point_type{X.x[ex][qx], Y.x[ey][qy]});
                        const double avx = (ux_v + conv * (ux_v * ux_x + uy_v * ux_y) + d * F[0]) * wj;
                        const double axx = (bx * ux_x + c * pv) * wj;
                        const double ayx = (by * ux_y) * wj;
                        const double avy = (uy_v + conv * (ux_v * uy_x + uy_v * uy_y) + d * F[1]) * wj;
                        const double axy = (bx * uy_x) * wj;
                        const double ayy = (by * uy_y + c * pv) * wj;
                        for (int b = 0; b < ndy; ++b) {
                            const double t1x = avx * yv[b] + ayx * yd[b];
                            const double t2x = axx * yv[b];
                            const double t1y = avy * yv[b] + ayy * yd[b];
                            const double t2y = axy * yv[b];
                            for (int a = 0; a < ndx; ++a) {
                                lx[b * 4 + a] += xv[a] * t1x + xd[a] * t2x;
                                ly[b * 4 + a] += xv[a] * t1y + xd[a] * t2y;
                            }
                        }
                    }
                }
                for (int b = 0; b < ndy; ++b) {
                    for (int a = 0; a < ndx; ++a) {
                        RX[(fx + a) + static_cast<std::size_t>(sv) * (fy + b)] += lx[b * 4 + a];
                        RY[(fx + a) + static_cast<std::size_t>(sv) * (fy + b)] += ly[b * 4 + a];
                    }
                }
            }
        });
    }

    void fill_sink_cache(double t) {
        const auto& PX = trial.Px.basis;
        const auto& PY = trial.Py.basis;
        const int nqx = PX.quad_order, nqy = PY.quad_order, ney = PY.elements;
        sink_cache.assign(static_cast<std::size_t>(PX.elements) * ney * nqx * nqy, 0.0);
        for_element_columns(PX.elements, PX.degree, [&](int ex) {
            for (int ey = 0; ey < ney; ++ey) {
                for (int qx = 0; qx < nqx; ++qx) {
                    for (int qy = 0; qy < nqy; ++qy) {
                        sink_cache[((static_cast<std::size_t>(ex) * ney + ey) * nqx + qx) * nqy + qy] =
                            problem.sink(point_type{PX.x[ex][qx], PY.x[ey][qy]}, t);
                    }
                }
            }
        });
    }

    void compute_rhs_pressure_1_fast(vector_type& rhs, double t) {
        const auto& X = trial.U1x.basis;
        const auto& Y = trial.U1y.basis;
        const auto& PX = trial.Px.basis;
        const auto& PY = trial.Py.basis;
        const int ndx = X.dofs_per_element(), ndy = Y.dofs_per_element();
        const int npx = PX.dofs_per_element(), npy = PY.dofs_per_element();
        const int nqx = X.quad_order, nqy = Y.quad_order;
        const int ney = Y.elements;
        const double dt = steps.dt;
        const int sv = trial.U1x.dofs(), sp = trial.Px.dofs();
        const double* UX = &vx(0, 0);
        const double* UY = &vy(0, 0);
        double* R = &rhs(0, 0);

        constexpr bool static_sink = detail::has_static_sink<Problem>::value;
        if constexpr (static_sink) {
            if (sink_cache.empty()) {
                fill_sink_cache(t);
            }
        }
        const double* SC = sink_cache.data();

        for_element_columns(X.elements, X.degree, [&](int ex) {
            for (int ey = 0; ey < ney; ++ey) {
                const int fx = X.first_dof(ex), fy = Y.first_dof(ey);
                const int pfx = PX.first_dof(ex), pfy = PY.first_dof(ey);
                double cxl[16], cyl[16];
                for (int b = 0; b < ndy; ++b) {
                    for (int a = 0; a < ndx; ++a) {
                        cxl[b * 4 + a] = UX[(fx + a) + static_cast<std::size_t>(sv) * (fy + b)];
                        cyl[b * 4 + a] = UY[(fx + a) + static_cast<std::size_t>(sv) * (fy + b)];
                    }
                }
                double loc[16] = {};
                const double J = X.J[ex] * Y.J[ey];
                for (int qx = 0; qx < nqx; ++qx) {
                    const double* xv = X.b[ex][qx][0];
                    const double* xd = X.b[ex][qx][1];
                    const double* pxv = PX.b[ex][qx][0];
                    double txd[4], txv[4];
                    for (int b = 0; b < ndy; ++b) {
                        double s0 = 0, s1 = 0;
                        for (int a = 0; a < ndx; ++a) {
                            s0 += cxl[b * 4 + a] * xd[a];
                            s1 += cyl[b * 4 + a] * xv[a];
                        }
                        txd[b] = s0;
                        txv[b] = s1;
                    }
                    for (int qy = 0; qy < nqy; ++qy) {
                        const double* yv = Y.b[ey][qy][0];
                        const double* yd = Y.b[ey][qy][1];
                        const double* pyv = PY.b[ey][qy][0];
                        double dvx = 0, dvy = 0;
                        for (int b = 0; b < ndy; ++b) {
                            dvx += txd[b] * yv[b];
                            dvy += txv[b] * yd[b];
                        }
                        double s;
                        if constexpr (static_sink) {
                            s = SC[((static_cast<std::size_t>(ex) * ney + ey) * nqx + qx) * nqy + qy];
                        } else {
                            s = problem.sink(point_type{PX.x[ex][qx], PY.x[ey][qy]}, t);
                        }
                        const double val = (-1 / dt * (dvx + dvy) - s / dt) * (X.w[qx] * Y.w[qy] * J);
                        for (int b = 0; b < npy; ++b) {
                            const double tb = val * pyv[b];
                            for (int a = 0; a < npx; ++a) {
                                loc[b * 4 + a] += tb * pxv[a];
                            }
                        }
                    }
                }
                for (int b = 0; b < npy; ++b) {
                    for (int a = 0; a < npx; ++a) {
                        R[(pfx + a) + static_cast<std::size_t>(sp) * (pfy + b)] += loc[b * 4 + a];
                    }
                }
            }
        });
    }

    double average_p_fast() {
        if (wpx.empty()) {
            auto integrals = [](const dimension& D) {
                std::vector<double> w(D.dofs(), 0.0);
                const auto& B = D.basis;
                for (int e = 0; e < B.elements; ++e) {
                    for (int q = 0; q < B.quad_order; ++q) {
                        for (int a = 0; a < B.dofs_per_element(); ++a) {
                            w[B.first_dof(e) + a] += B.b[e][q][0][a] * B.w[q] * B.J[e];
                        }
                    }
                }
                return w;
            };
            wpx = integrals(trial.Px);
            wpy = integrals(trial.Py);
        }
        double val = 0;
        for (int j = 0; j < trial.Py.dofs(); ++j) {
            double s = 0;
            for (int i = 0; i < trial.Px.dofs(); ++i) {
                s += p(i, j) * wpx[i];
            }
            val += s * wpy[j];
        }
        return val / ((trial.Px.b - trial.Px.a) * (trial.Py.b - trial.Py.a));
    }

    void update_velocity_kron(double t) {
        prepare_kron();
        const auto dt = steps.dt;
        auto f = [&](point_type x, double s) { return problem.forcing(x, s); };
        auto F = [&](double s) { return [&, s](point_type x) { return f(x, s); }; };
        const auto Re = problem.Re;
        const auto conv = problem.navier_stokes ? dt / 2 : 0;

        using shape = std::array<int, 2>;
        const shape s1{trial.U1x.dofs(), trial.U1y.dofs()};
        const shape s2{trial.U2x.dofs(), trial.U2y.dofs()};
        vector_type rx{s1}, ry{s2}, gx{s1}, gy{s2}, bx{s1}, by{s2}, sx{s1}, sy{s2};

        {
            detail::scoped_acc t_rhs{timing.rhs};
            {
                detail::scoped_acc t_p{timing.rhs_v1};
                if (use_fast) {
                    compute_rhs_fast(rx, ry, vx, vy, p_star, F(t + dt / 2), 0, -dt / (2 * Re), -conv,
                                     dt / 2, dt / 2);
                } else {
                    compute_rhs(rx, ry, vx, vy, vx, vy, p_star, F(t + dt / 2), 0, 0, 0,
                                -dt / (2 * Re), -conv, dt / 2, dt / 2);
                }
            }
            detail::scoped_acc t_b{timing.rhs_bc};
            apply_velocity_bc(gx, trial.U1x, trial.U1y, t + dt, 0);
            apply_velocity_bc(gy, trial.U2x, trial.U2y, t + dt, 1);
        }
        {
            detail::scoped_acc t_k{timing.kron};
            if (use_fast && par_solve) {
                kron::solve_dirichlet_par(executor, *kc.vel[0][0][1], *kc.vel[0][1][0], rx, gx, bx, sx);
                kron::solve_dirichlet_par(executor, *kc.vel[1][0][1], *kc.vel[1][1][0], ry, gy, by, sy);
            } else {
                kron::solve_dirichlet(*kc.vel[0][0][1], *kc.vel[0][1][0], rx, gx, bx, sx);
                kron::solve_dirichlet(*kc.vel[1][0][1], *kc.vel[1][1][0], ry, gy, by, sy);
            }
        }

        vector_type rx2{s1}, ry2{s2};
        {
            detail::scoped_acc t_rhs{timing.rhs};
            detail::scoped_acc t_p{timing.rhs_v2};
            if (use_fast) {
                compute_rhs_fast(rx2, ry2, rx, ry, p_star, F(t + dt / 2), -dt / (2 * Re), 0, -conv,
                                 dt / 2, dt / 2);
            } else {
                compute_rhs(rx2, ry2, vx, vy, rx, ry, p_star, F(t + dt / 2), 0, 0, -dt / (2 * Re), 0,
                            -conv, dt / 2, dt / 2);
            }
        }
        {
            detail::scoped_acc t_k{timing.kron};
            if (use_fast && par_solve) {
                kron::solve_dirichlet_par(executor, *kc.vel[0][0][0], *kc.vel[0][1][1], rx2, gx, bx, sx);
                kron::solve_dirichlet_par(executor, *kc.vel[1][0][0], *kc.vel[1][1][1], ry2, gy, by, sy);
            } else {
                kron::solve_dirichlet(*kc.vel[0][0][0], *kc.vel[0][1][1], rx2, gx, bx, sx);
                kron::solve_dirichlet(*kc.vel[1][0][0], *kc.vel[1][1][1], ry2, gy, by, sy);
            }
        }

        vx_prev = vx;
        vy_prev = vy;
        for (auto i : dofs(trial.U1x, trial.U1y)) {
            vx(i[0], i[1]) = rx2(i[0], i[1]);
        }
        for (auto i : dofs(trial.U2x, trial.U2y)) {
            vy(i[0], i[1]) = ry2(i[0], i[1]);
        }
    }

    void update_pressure_kron(double t) {
        prepare_kron();
        using shape = std::array<int, 2>;
        const shape sp{trial.Px.dofs(), trial.Py.dofs()};
        vector_type r{sp}, buf{sp};
        {
            detail::scoped_acc t_rhs{timing.rhs};
            detail::scoped_acc t_p{timing.rhs_p};
            if (use_fast) {
                compute_rhs_pressure_1_fast(r, t);
            } else {
                compute_rhs_pressure_1(r, vx, vy, test.Px, test.Py, steps.dt, t);
            }
        }
        {
            detail::scoped_acc t_k{timing.kron};
            if (use_fast && par_solve) {
                kron::solve_free_par(executor, *kc.pres[0], *kc.pres[1], r, buf);
            } else {
                kron::solve_free(*kc.pres[0], *kc.pres[1], r, buf);
            }
        }
        for (auto i : dofs(trial.Px, trial.Py)) {
            phi(i[0], i[1]) = r(i[0], i[1]);
        }
        for (auto i : dofs(trial.Px, trial.Py)) {
            p(i[0], i[1]) += phi(i[0], i[1]);
        }
    }

    struct kron_verify_t {
        double rhs_v1 = 0, rhs_v2 = 0, rhs_p = 0;
        double step_vx = 0, step_vy = 0, step_p = 0, step_phi = 0;
        double rep_generic_rhs = 0;  // compute_rhs (locks, thread-order sums) run twice: relative difference
        double res_kron = 0, res_mumps = 0;  // residual of (A1 (x) A1) phi = r_p (r_p from the velocity after the step of that solver)
        double rep_kron = 0, rep_mumps = 0, rep_m_vx = 0, rep_m_vy = 0, rep_m_p = 0, rep_m_phi = 0;  // repeatability of each solver (same state, step repeated); information only
        double worst_rhs() const { return std::max({rhs_v1, rhs_v2, rhs_p}); }
        double worst_step() const { return std::max({step_vx, step_vy, step_p, step_phi}); }
    };

    kron_verify_t verify_kron(bool force_parallel) {
        if (!use_kron || !use_fast) {
            throw std::runtime_error("verify_kron: needs the Kronecker solver with fast kernels");
        }
        if (!verify_ready) {
            before();
            verify_ready = true;
        }
        const bool keep_rhs = par_rhs, keep_solve = par_solve;
        if (force_parallel) {
            par_rhs = true;
            par_solve = true;
        }
        auto rel = [](const vector_type& a, const vector_type& b) {
            double num = 0, den = 0;
            for (std::size_t k = 0; k < static_cast<std::size_t>(b.size()); ++k) {
                num = std::max(num, std::abs(a.data()[k] - b.data()[k]));
                den = std::max(den, std::abs(b.data()[k]));
            }
            return den > 0 ? num / den : num;
        };
        kron_verify_t res;
        const double t = 0.0;
        const double dt = steps.dt, Re = problem.Re;
        const double conv = problem.navier_stokes ? dt / 2 : 0;
        auto F = [&](double s) { return [&, s](point_type x) { return problem.forcing(x, s); }; };
        using shape = std::array<int, 2>;
        const shape s1{trial.U1x.dofs(), trial.U1y.dofs()};
        const shape s2{trial.U2x.dofs(), trial.U2y.dofs()};
        const shape sp{trial.Px.dofs(), trial.Py.dofs()};

        const vector_type save_vx = vx, save_vy = vy, save_p = p, save_phi = phi, save_vxp = vx_prev,
                          save_vyp = vy_prev;
        auto restore = [&]() {
            vx = save_vx;
            vy = save_vy;
            p = save_p;
            phi = save_phi;
            vx_prev = save_vxp;
            vy_prev = save_vyp;
        };

        compute_pressure_predictor();
        vector_type fx{s1}, fy{s2}, gx{s1}, gy{s2};
        compute_rhs_fast(fx, fy, vx, vy, p_star, F(t + dt / 2), 0, -dt / (2 * Re), -conv, dt / 2, dt / 2);
        compute_rhs(gx, gy, vx, vy, vx, vy, p_star, F(t + dt / 2), 0, 0, 0, -dt / (2 * Re), -conv,
                    dt / 2, dt / 2);
        res.rhs_v1 = std::max(rel(fx, gx), rel(fy, gy));
        {
            vector_type hx{s1}, hy{s2};
            compute_rhs(hx, hy, vx, vy, vx, vy, p_star, F(t + dt / 2), 0, 0, 0, -dt / (2 * Re), -conv,
                        dt / 2, dt / 2);
            res.rep_generic_rhs = std::max(rel(hx, gx), rel(hy, gy));
        }

        prepare_kron();
        vector_type bcx{s1}, bcy{s2}, bx{s1}, by{s2}, sx{s1}, sy{s2}, ix = gx, iy = gy;
        apply_velocity_bc(bcx, trial.U1x, trial.U1y, t + dt, 0);
        apply_velocity_bc(bcy, trial.U2x, trial.U2y, t + dt, 1);
        kron::solve_dirichlet(*kc.vel[0][0][1], *kc.vel[0][1][0], ix, bcx, bx, sx);
        kron::solve_dirichlet(*kc.vel[1][0][1], *kc.vel[1][1][0], iy, bcy, by, sy);
        vector_type fx2{s1}, fy2{s2}, gx2{s1}, gy2{s2};
        compute_rhs_fast(fx2, fy2, ix, iy, p_star, F(t + dt / 2), -dt / (2 * Re), 0, -conv, dt / 2, dt / 2);
        compute_rhs(gx2, gy2, vx, vy, ix, iy, p_star, F(t + dt / 2), 0, 0, -dt / (2 * Re), 0, -conv,
                    dt / 2, dt / 2);
        res.rhs_v2 = std::max(rel(fx2, gx2), rel(fy2, gy2));


        step(0, t);
        const vector_type k_vx = vx, k_vy = vy, k_p = p, k_phi = phi;
        auto pres_residual = [&]() {
            vector_type r{sp}, out{sp};
            compute_rhs_pressure_1(r, vx, vy, test.Px, test.Py, dt, t);
            kron::apply(*kc.pres[0], *kc.pres[1], phi, out);
            return rel(out, r);
        };
        res.res_kron = pres_residual();

        vector_type fp{sp}, gp{sp};
        sink_cache.clear();
        compute_rhs_pressure_1_fast(fp, t);
        compute_rhs_pressure_1(gp, vx, vy, test.Px, test.Py, dt, t);
        res.rhs_p = rel(fp, gp);

        restore();
        use_kron = false;
        step(0, t);
        use_kron = true;
        res.res_mumps = pres_residual();
        res.step_vx = rel(k_vx, vx);
        res.step_vy = rel(k_vy, vy);
        res.step_p = rel(k_p, p);
        res.step_phi = rel(k_phi, phi);
        const vector_type m_vx = vx, m_vy = vy, m_p = p, m_phi = phi;

        restore();
        use_kron = false;
        step(0, t);
        use_kron = true;
        res.rep_m_vx = rel(vx, m_vx);
        res.rep_m_vy = rel(vy, m_vy);
        res.rep_m_p = rel(p, m_p);
        res.rep_m_phi = rel(phi, m_phi);
        res.rep_mumps = std::max({rel(vx, m_vx), rel(vy, m_vy), rel(p, m_p), rel(phi, m_phi)});
        restore();
        step(0, t);
        res.rep_kron = std::max({rel(vx, k_vx), rel(vy, k_vy), rel(p, k_p), rel(phi, k_phi)});

        restore();
        par_rhs = keep_rhs;
        par_solve = keep_solve;
        return res;
    }

    void step(int iter, double t) override {
        detail::scoped_acc t_step{timing.steps};
        compute_pressure_predictor();

        if (use_kron) {
            update_velocity_kron(t);
            update_pressure_kron(t);
            return;
        }

        update_velocity_igrm(iter, t);

        update_pressure_igrm(t);
    }

    template <typename RHS, typename S1, typename S2, typename S3, typename Fun>
    void compute_rhs(RHS& rhsx, RHS& rhsy, const S1& vx0, const S1& vy0, const S2& vx, const S2& vy,
                     const S3& p, Fun&& forcing, double ax, double ay, double bx, double by,
                     double conv, double c, double d) const {
        using shape = std::array<int, 2>;
        auto u1_shape = shape{test.U1x.basis.dofs_per_element(), test.U1y.basis.dofs_per_element()};
        auto u2_shape = shape{test.U2x.basis.dofs_per_element(), test.U2y.basis.dofs_per_element()};

        executor.for_each(elements(trial.Px, trial.Py), [&](index_type e) {
            auto vx_loc = vector_type{u1_shape};
            auto vy_loc = vector_type{u2_shape};

            double J = jacobian(e);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double W = weight(q);
                auto x = point(e, q);
                auto F = forcing(x);
                value_type vvx0 = eval(vx0, e, q, trial.U1x, trial.U1y);
                value_type vvy0 = eval(vy0, e, q, trial.U2x, trial.U2y);
                value_type vvx = eval(vx, e, q, trial.U1x, trial.U1y);
                value_type vvy = eval(vy, e, q, trial.U2x, trial.U2y);
                value_type pp = eval(p, e, q, trial.Px, trial.Py);

                for (auto a : dofs_on_element(e, test.U1x, test.U1y)) {
                    auto aa = dof_global_to_local(e, a, test.U1x, test.U1y);
                    value_type v = eval_basis(e, q, a, test.U1x, test.U1y);

                    double val = vvx.val * v.val
                               + ax * vvx0.dx * v.dx
                               + ay * vvx0.dy * v.dy
                               + bx * vvx.dx * v.dx
                               + by * vvx.dy * v.dy
                               + c * pp.val * v.dx
                               + conv * (vvx.val * vvx.dx + vvy.val * vvx.dy) * v.val
                               + d * F[0] * v.val;
                    vx_loc(aa[0], aa[1]) += val * W * J;
                }
                for (auto a : dofs_on_element(e, test.U2x, test.U2y)) {
                    auto aa = dof_global_to_local(e, a, test.U2x, test.U2y);
                    value_type v = eval_basis(e, q, a, test.U2x, test.U2y);

                    double val = vvy.val * v.val
                               + ax * vvy0.dx * v.dx
                               + ay * vvy0.dy * v.dy
                               + bx * vvy.dx * v.dx
                               + by * vvy.dy * v.dy
                               + c * pp.val * v.dy
                               + conv * (vvx.val * vvy.dx + vvy.val * vvy.dy) * v.val
                               + d * F[1] * v.val;
                    vy_loc(aa[0], aa[1]) += val * W * J;
                }
            }
            executor.synchronized([&]() {
                update_global_rhs(rhsx, vx_loc, e, test.U1x, test.U1y);
                update_global_rhs(rhsy, vy_loc, e, test.U2x, test.U2y);
            });
        });
    }

    template <typename RHS, typename Sol>
    void compute_rhs_pressure_1(RHS& rhs, const Sol& vx, const Sol& vy, const dimension& Vx,
                                const dimension& Vy, double dt, double t) const {
        using shape = std::array<int, 2>;
        auto p_shape = shape{Vx.basis.dofs_per_element(), Vy.basis.dofs_per_element()};

        executor.for_each(elements(trial.Px, trial.Py), [&](index_type e) {
            auto loc = vector_type{p_shape};

            double J = jacobian(e);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double W = weight(q);
                auto x = point(e, q);
                value_type vvx = eval(vx, e, q, trial.U1x, trial.U1y);
                value_type vvy = eval(vy, e, q, trial.U2x, trial.U2y);
                double s = problem.sink(x, t);

                for (auto a : dofs_on_element(e, Vx, Vy)) {
                    auto aa = dof_global_to_local(e, a, Vx, Vy);
                    value_type v = eval_basis(e, q, a, Vx, Vy);

                    double val = (-1 / dt * (vvx.dx + vvy.dy) - s / dt) * v.val;
                    loc(aa[0], aa[1]) += val * W * J;
                }
            }
            executor.synchronized([&]() { update_global_rhs(rhs, loc, e, Vx, Vy); });
        });
    }

    template <typename RHS, typename Sol>
    void compute_rhs_pressure_2(RHS& rhs, const Sol& p, const dimension& Vx,
                                const dimension& Vy) const {
        using shape = std::array<int, 2>;
        auto p_shape = shape{Vx.basis.dofs_per_element(), Vy.basis.dofs_per_element()};

        executor.for_each(elements(trial.Px, trial.Py), [&](index_type e) {
            auto loc = vector_type{p_shape};

            double J = jacobian(e);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double W = weight(q);
                value_type pp = eval(p, e, q, trial.Px, trial.Py);

                for (auto a : dofs_on_element(e, Vx, Vy)) {
                    auto aa = dof_global_to_local(e, a, Vx, Vy);
                    value_type v = eval_basis(e, q, a, Vx, Vy);

                    double val = pp.val * v.val;
                    loc(aa[0], aa[1]) += val * W * J;
                }
            }
            executor.synchronized([&]() { update_global_rhs(rhs, loc, e, Vx, Vy); });
        });
    }

    template <typename RHS>
    void compute_rhs_pressure_update(RHS& rhs, double chi) const {
        auto Re = problem.Re;

        using shape = std::array<int, 2>;
        auto p_shape = shape{trial.Px.basis.dofs_per_element(), trial.Py.basis.dofs_per_element()};

        executor.for_each(elements(trial.Px, trial.Py), [&](index_type e) {
            auto loc = vector_type{p_shape};

            double J = jacobian(e);
            for (auto q : quad_points(trial.Px, trial.Py)) {
                double W = weight(q);
                value_type pp = eval(p, e, q, trial.Px, trial.Py);
                value_type pphi = eval(phi, e, q, trial.Px, trial.Py);
                value_type vvx = eval(vx, e, q, trial.U1x, trial.U1y);
                value_type vvy = eval(vy, e, q, trial.U2x, trial.U2y);
                value_type vvx_prev = eval(vx_prev, e, q, trial.U1x, trial.U1y);
                value_type vvy_prev = eval(vy_prev, e, q, trial.U2x, trial.U2y);

                for (auto a : dofs_on_element(e, trial.Px, trial.Py)) {
                    auto aa = dof_global_to_local(e, a, trial.Px, trial.Py);
                    value_type v = eval_basis(e, q, a, trial.Px, trial.Py);

                    double vdiv = vvx.dx + vvy.dy;
                    double vdiv_prev = vvx_prev.dx + vvy_prev.dy;
                    double val = (pp.val + pphi.val - 0.5 * chi / Re * (vdiv + vdiv_prev)) * v.val;
                    loc(aa[0], aa[1]) += val * W * J;
                }
            }
            executor.synchronized([&]() { update_global_rhs(rhs, loc, e, trial.Px, trial.Py); });
        });
    }

    void after_step(int iter, double t) override {
        detail::scoped_acc t_after{timing.after_step};
        int i = iter + 1;
        double tt = t + steps.dt;
        double th = t + steps.dt / 2;

        auto e_vx = [this, tt](point_type x) { return problem.exact_v(x, tt)[0]; };
        auto e_vy = [this, tt](point_type x) { return problem.exact_v(x, tt)[1]; };
        auto e_p = [this, th](point_type x) { return problem.exact_p(x, th); };

        {
            detail::scoped_acc t_shift{timing.shift};
            double p_avg;
            if (use_kron && use_fast) {
                p_avg = average_p_fast();
            } else {
                p_avg = average_value(p, trial.Px, trial.Py);
            }
            double p_exact_avg = 0.0;
            if constexpr (!detail::has_zero_exact_pressure<Problem>::value) {
                p_exact_avg = average_value(trial.Px, trial.Py, e_p);
            }
            shift_pressure(p, p_exact_avg - p_avg);
        }

        if (save_every_step && i % 1 == 0) {
            save_to_file(i);

        }


        const long gs = global_step++;
        if (diag_every > 1 && gs % diag_every != 0) {
            return;
        }

        double vx_norm_L2 = normL2(vx, trial.U1x, trial.U1y);
        double vy_norm_L2 = normH1(vy, trial.U2x, trial.U2y);
        double v_norm_L2 = std::hypot(vx_norm_L2, vy_norm_L2);

        double vx_norm_H1 = normL2(vx, trial.U1x, trial.U1y);
        double vy_norm_H1 = normH1(vy, trial.U2x, trial.U2y);
        double v_norm_H1 = std::hypot(vx_norm_H1, vy_norm_H1);

        double p_norm_L2 = normL2(p, trial.Px, trial.Py);
        double phi_norm_L2 = normL2(phi, trial.Px, trial.Py);

        double vx_error_L2 = error_relative_L2(vx, trial.U1x, trial.U1y, e_vx) * 100;
        double vy_error_L2 = error_relative_L2(vy, trial.U2x, trial.U2y, e_vy) * 100;
        double v_error_L2 = std::hypot(vx_error_L2, vy_error_L2);

        double vx_error_H1 = error_relative_H1(vx, trial.U1x, trial.U1y, e_vx) * 100;
        double vy_error_H1 = error_relative_H1(vy, trial.U2x, trial.U2y, e_vy) * 100;
        double v_error_H1 = std::hypot(vx_error_H1, vy_error_H1);

        double p_error_L2 = error_relative_L2(p, trial.Px, trial.Py, e_p) * 100;

        std::cout << i << " " << tt                                  //
                  << " |v| = " << v_norm_L2 << " " << v_norm_H1      //
                  << " |p| = " << p_norm_L2                          //
                  << " v err = " << v_error_L2 << " " << v_error_H1  //
                  << " vx_err = " << vx_error_L2                     //
                  << " vy_err = " << vy_error_L2                     //
                  << " p err = " << p_error_L2                       //
                  << " |phi| = " << phi_norm_L2                      //
                  << std::endl;

    }

    void after() override {
        total_timer.stop();
        auto time = static_cast<double>(total_timer.get()) / 1000.0;
        std::cout << "Total time: " << time << " s" << std::endl;
    }

    template <typename Sol>
    void shift_pressure(Sol& pressure, double delta) const {
        for (auto i : dofs(trial.Px, trial.Py)) {
            pressure(i[0], i[1]) += delta;
        }
    }

    template <typename Sol>
    double average_value(const Sol& u, const dimension& Ux, const dimension& Uy) const {
        double val = 0;
        for (auto e : elements(Ux, Uy)) {
            double J = jacobian(e, Ux, Uy);
            for (auto q : quad_points(Ux, Uy)) {
                double w = weight(q, Ux, Uy);
                value_type uu = eval(u, e, q, Ux, Uy);
                val += uu.val * w * J;
            }
        }
        double area = (Ux.b - Ux.a) * (Uy.b - Uy.a);
        return val / area;
    }

    template <typename Fun>
    double average_value(const dimension& Ux, const dimension& Uy, Fun&& fun) const {
        double val = 0;
        for (auto e : elements(Ux, Uy)) {
            double J = jacobian(e, Ux, Uy);
            for (auto q : quad_points(Ux, Uy)) {
                double w = weight(q, Ux, Uy);
                auto x = point(e, q, Ux, Uy);
                auto fx = fun(x);
                val += fx.val * w * J;
            }
        }
        double area = (Ux.b - Ux.a) * (Uy.b - Uy.a);
        return val / area;
    }

    void save_to_file(int i) {
        outputP.to_file(p, "pressure_%d.data", i);
        outputU1.to_file(vx, "vx_%d.data", i);
        outputU2.to_file(vy, "vy_%d.data", i);
    }

    template <typename RHS>
    void zero_bc(RHS& u, dimension& Ux, dimension& Uy) const {
        for_boundary_dofs(Ux, Uy, [&](index_type i) { u(i[0], i[1]) = 0; });
    }

    template <typename Sol, typename Fun>
    double div_errorL2(const Sol& u, const Sol& v, const space_set& space, Fun&& fun) const {
        auto L2 = [](value_type a) { return a.val * a.val; };
        return div_error(u, v, space, L2, fun);
    }

    template <typename Sol, typename Fun>
    double div_errorH1(const Sol& u, const Sol& v, const space_set& space, Fun&& fun) const {
        auto H1 = [](value_type a) { return a.val * a.val + a.dx * a.dx + a.dy * a.dy; };
        return div_error(u, v, space, H1, fun);
    }

    template <typename Sol, typename Fun, typename Norm>
    double div_error(const Sol& u, const Sol& v, const space_set& space, Norm&& norm,
                     Fun&& fun) const {
        double error = 0;

        for (auto e : elements(space.Px, space.Px)) {
            double J = jacobian(e, space.Px, space.Py);
            for (auto q : quad_points(space.Px, space.Py)) {
                double w = weight(q, space.Px, space.Py);
                auto x = point(e, q, space.Px, space.Py);
                value_type div = divergence(u, v, e, q, space);

                auto d = div - fun(x);
                error += norm(d) * w * J;
            }
        }
        return std::sqrt(error);
    }

    template <typename Sol>
    value_type divergence(const Sol& u, const Sol& v, index_type e, index_type q,
                          const space_set& space) const {
        value_type div{};
        for (auto b : dofs_on_element(e, space.U1x, space.U1y)) {
            double c = u(b[0], b[1]);

            auto loc = dof_global_to_local(e, b, space.U1x, space.U1y);

            const auto& bx = space.U1x.basis;
            const auto& by = space.U1y.basis;

            double B2 = by.b[e[1]][q[1]][0][loc[1]];
            double dB1 = bx.b[e[0]][q[0]][1][loc[0]];
            double dB2 = by.b[e[1]][q[1]][1][loc[1]];
            double ddB1 = bx.b[e[0]][q[0]][2][loc[0]];

            double dx = dB1 * B2;
            double dxx = ddB1 * B2;
            double dxy = dB1 * dB2;

            div.val += c * dx;
            div.dx += c * dxx;
            div.dy += c * dxy;
        }
        for (auto b : dofs_on_element(e, space.U2x, space.U2y)) {
            double d = v(b[0], b[1]);

            auto loc = dof_global_to_local(e, b, space.U2x, space.U2y);

            const auto& bx = space.U2x.basis;
            const auto& by = space.U2y.basis;

            double B1 = bx.b[e[0]][q[0]][0][loc[0]];
            double dB1 = bx.b[e[0]][q[0]][1][loc[0]];
            double dB2 = by.b[e[1]][q[1]][1][loc[1]];
            double ddB2 = by.b[e[1]][q[1]][2][loc[1]];

            double dy = B1 * dB2;
            double dyy = B1 * ddB2;
            double dxy = dB1 * dB2;

            div.val += d * dy;
            div.dx += d * dxy;
            div.dy += d * dyy;
        }
        return div;
    }

    bool overlap(int a, const dimension& U, int b, const dimension& V) const {
        auto ar = U.basis.element_ranges[a];
        auto br = V.basis.element_ranges[b];
        return (ar.first >= br.first && ar.first <= br.second)
            || (br.first >= ar.first && br.first <= ar.second);
    }

    bool overlap(index_type a, const dimension& Ux, const dimension& Uy, index_type b,
                 const dimension& Vx, const dimension& Vy) const {
        return overlap(a[0], Ux, b[0], Vx) && overlap(a[1], Uy, b[1], Vy);
    }

    template <typename Form>
    double integrate(index_type i, index_type j, const dimension& Ux, const dimension& Uy,
                     const dimension& Vx, const dimension& Vy, Form&& form) const {
        double val = 0;

        for (auto e : elements_supporting_dof(i, Ux, Uy)) {
            if (!supported_in(j, e, Vx, Vy))
                continue;

            double J = jacobian(e, Ux, Uy);
            for (auto q : quad_points(Ux, Uy)) {
                double w = weight(q);
                value_type ww = eval_basis(e, q, i, Ux, Uy);
                value_type uu = eval_basis(e, q, j, Vx, Vy);

                double fuw = form(ww, uu);
                val += fuw * w * J;
            }
        }
        return val;
    }
};

}

#endif
