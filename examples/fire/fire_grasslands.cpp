// SPDX-FileCopyrightText: 2015 - 2024 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT
// Grasslands National Park fire simulation

#include <map>

#include "fire_grasslands.hpp"

namespace {


struct cli_options {
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

int main(int argc_in, char* argv_in[]) {
    std::vector<std::string> args;
    cli_options opt;
    for (int i = 1; i < argc_in; ++i) {
        std::string a = argv_in[i];
        if (a.rfind("--", 0) == 0) {
            auto eq = a.find('=');
            if (eq == std::string::npos) {
                opt.kv[a.substr(2)] = "1";
            } else {
                opt.kv[a.substr(2, eq - 2)] = a.substr(eq + 1);
            }
        } else {
            args.push_back(a);
        }
    }

    std::vector<const char*> argv_v{argv_in[0]};
    for (auto& a : args) {
        argv_v.push_back(a.c_str());
    }
    int argc = static_cast<int>(argv_v.size());
    const char** argv = argv_v.data();

    if (argc < 5) {
        std::cerr << "Usage: fire_grasslands <Nx> <Ny> <p> <threads> [fuel_map_file] [ignition_x] [ignition_y] [n_steps] [dt] [wind_x] [wind_y] [coupling_step] [kb] [period] [max_cycles] [recenter] [search_radius_m] [--options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Parameters:" << std::endl;
        std::cerr << "  Nx             - mesh size in x direction (e.g., 400)" << std::endl;
        std::cerr << "  Ny             - mesh size in y direction (e.g., 200)" << std::endl;
        std::cerr << "  p              - polynomial degree (e.g., 2)" << std::endl;
        std::cerr << "  threads        - number of threads" << std::endl;
        std::cerr << "  fuel_map_file  - path to fuel map data file (optional)" << std::endl;
        std::cerr << "  ignition_x     - x coordinate of ignition point (default: domain_x/2)" << std::endl;
        std::cerr << "  ignition_y     - y coordinate of ignition point (default: domain_y/2)" << std::endl;
        std::cerr << "  n_steps        - number of time steps (default: 1000)" << std::endl;
        std::cerr << "  dt             - time step size (default: 1e-4)" << std::endl;
        std::cerr << "  wind_x         - background wind in x direction, in m/s (default: 15.0)" << std::endl;
        std::cerr << "  wind_y         - background wind in y direction, in m/s (default: 0.0)" << std::endl;
        std::cerr << "  coupling_step  - first N-S coupling cycle at this step (<=0 or omitted: no coupling)" << std::endl;
        std::cerr << "  kb             - sink closure kb (<=0: 0.001563)" << std::endl;
        std::cerr << "  period         - recouple every `period` steps (0: one-shot)" << std::endl;
        std::cerr << "  max_cycles     - number of coupling cycles (<=0 or omitted: unlimited, v3 K6)" << std::endl;
        std::cerr << "  recenter       - 0/1: recenter the 1 km window on the hottest point" << std::endl;
        std::cerr << "  search_radius_m- recentering search radius in meters" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options (v3):" << std::endl;
        std::cerr << "  --ignition-r=<r> --ignition-R=<R>  ignition bump radii in sim-units (default 0.5 2.0)" << std::endl;
        std::cerr << "  --scale-x-m=<m> --scale-y-m=<m>    meters per sim-unit (default 217.7 310.8)" << std::endl;
        std::cerr << "  --fire-time-s=<s>                  seconds per unit of fire time (default 1, unverified)" << std::endl;
        std::cerr << "  --domain-x=<sim-units>             domain width (default 100); domain_y = domain_x*Ny/Nx (v6 2a)" << std::endl;        std::cerr << "  --ns-sink-law=legacy|rate|fuel     S = kb*dr/h (legacy) | kb_phys*(dr/t_u)/h | kb_phys*C_F*dr*F/(t_u*h) (v6 2b)" << std::endl;
        std::cerr << "  --ns-kb-phys=<v>                   kb_phys for rate/fuel (default kb*t_u; fuel: kb*t_u/C_F, C_F=300)" << std::endl;
        std::cerr << "  --ns-source=rate|temperature       field handed to the wind solver (default rate)" << std::endl;
        std::cerr << "  --ns-domain=window|full|adaptive   N-S domain (default window = 1 km square)" << std::endl;
        std::cerr << "  --ns-window-m --ns-elem-m --ns-margin-m --ns-min-window-m --ns-sample-m" << std::endl;
        std::cerr << "  --ns-N=<elements> --ns-nsteps=<n> --ns-t-end=<T units> --ns-time=fixed|sync" << std::endl;
        std::cerr << "  --ns-Re=<Re> | --ns-nu=<m^2/s>     (Re = 1e6/nu for L=1000 m, T=1 s)" << std::endl;
        std::cerr << "  --ignition-line (ignite along x = ignition_x)  --cw=<value> (constant cw, default 0.5)" << std::endl;
        std::cerr << "  --qw-back-table=u0:b0,u1:b1,...  (backing-fire correction of Qw<0)" << std::endl;
        std::cerr << "  --cw-law=const|grasso|linear (--cw-a, --cw-b)  --out-every=<steps>" << std::endl;
        std::cerr << "  --cw-law=const|grasso  --qw-supg   (diagnostics of the wind transport term Qw)" << std::endl;
        std::cerr << "  --ns-legacy (pre-v3 coupling: temperature input, 1 km window, cold, fixed horizon, v=b)" << std::endl;
        std::cerr << "  --ns-solver=kron|mumps (default kron: Kronecker-product ADS solves; mumps: sparse direct, the pre-1f solver)" << std::endl;
        std::cerr << "  --ns-warm|--ns-cold (default warm) --ns-bc=uniform|farfield --ns-fields-n=<n> --ns-no-fields --ns-fields-every=<n> --ns-inflow-diag --ns-no-keep-io" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  ./fire_grasslands 400 200 2 4 grasslands_fuel.data 50 25 2000 1e-4 0.0 0.0 200 -1 100 50 1 1500" << std::endl;
        std::exit(1);
    }

    int nx = std::atoi(argv[1]);
    int ny = std::atoi(argv[2]);
    int p = std::atoi(argv[3]);
    int threads = std::atoi(argv[4]);

    std::string fuel_map_file = "";
    if (argc >= 6) {
        fuel_map_file = argv[5];
    }


    double domain_x = opt.num("domain-x", 100.0);
    double domain_y = domain_x * ny / nx;


    double ignition_x = domain_x / 2.0;
    double ignition_y = domain_y / 2.0;
    if (argc >= 8) {
        ignition_x = std::atof(argv[6]);
        ignition_y = std::atof(argv[7]);
    }


    int n_steps = 1000;
    double dt = 1e-4;
    if (argc >= 9) {
        n_steps = std::atoi(argv[8]);
    }
    if (argc >= 10) {
        dt = std::atof(argv[9]);
    }


    double wind_x = 15.0;
    double wind_y = 0.0;
    if (argc >= 11) {
        wind_x = std::atof(argv[10]);
    }
    if (argc >= 12) {
        wind_y = std::atof(argv[11]);
    }


    int coupling_step = -1;
    if (argc >= 13) {
        coupling_step = std::atoi(argv[12]);
    }

    double coupling_kb_override = -1.0;
    if (argc >= 14) {
        coupling_kb_override = std::atof(argv[13]);
    }


    int coupling_period = 0;
    if (argc >= 15) {
        coupling_period = std::atoi(argv[14]);
    }
    int coupling_max_cycles = -1;
    if (argc >= 16) {
        coupling_max_cycles = std::atoi(argv[15]);
    }
    bool coupling_recenter = false;
    if (argc >= 17) {
        coupling_recenter = std::atoi(argv[16]) != 0;
    }
    double coupling_search_radius_m = 150.0;
    if (argc >= 18) {
        coupling_search_radius_m = std::atof(argv[17]);
    }

    double ignition_r = opt.num("ignition-r", 0.5);
    double ignition_R = opt.num("ignition-R", 2.0);
    double scale_x_m = opt.num("scale-x-m", 217.7);
    double scale_y_m = opt.num("scale-y-m", 310.8);
    double fire_time_s = opt.num("fire-time-s", 1.0);

    std::cout << "========================================" << std::endl;
    std::cout << "Grasslands National Park Fire Simulation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Mesh size: " << nx << " x " << ny << std::endl;
    std::cout << "  Domain: [0, " << domain_x << "] x [0, " << domain_y << "]" << std::endl;
    std::cout << "  Polynomial degree: " << p << std::endl;
    std::cout << "  Threads: " << threads << std::endl;
    if (!fuel_map_file.empty()) {
        std::cout << "  Fuel map: " << fuel_map_file << std::endl;
    } else {
        std::cout << "  Fuel map: uniform (default)" << std::endl;
    }
    std::cout << "  Ignition point: (" << ignition_x << ", " << ignition_y << ") radii "
              << ignition_r << " / " << ignition_R << std::endl;
    std::cout << "  Time steps: " << n_steps << " x " << dt << " = " << (n_steps * dt) << "s" << std::endl;
    std::cout << "  Wind: (" << wind_x << ", " << wind_y << ") m/s" << std::endl;
    std::cout << "  Scales: " << scale_x_m << " x " << scale_y_m << " m/sim-unit, fire time unit "
              << fire_time_s << " s" << std::endl;
    std::cout << "========================================" << std::endl;

    // domain configuration
    ads::dim_config dim_x{p, nx, 0.0, domain_x};
    ads::dim_config dim_y{p, ny, 0.0, domain_y};

    // time stepping configuration
    ads::timesteps_config steps{n_steps, dt};
    int ders = 1;

    ads::config_2d c{dim_x, dim_y, steps, ders};
    ads::problems::fire_grasslands sim{c, threads, fuel_map_file};

    // set domain dimensions for fuel map interpolation
    sim.set_domain(domain_x, domain_y);
    sim.set_scales(scale_x_m, scale_y_m);
    sim.set_fire_time_unit(fire_time_s);
    sim.set_wind_transport_options(opt.str("cw-law", "const"), opt.has("qw-supg"), domain_x / nx,
                                   domain_y / ny);


    sim.set_ignition_point(ignition_x, ignition_y, ignition_r, ignition_R);

    sim.set_ignition_line(opt.has("ignition-line"));
    if (opt.has("cw")) {
        sim.set_cw(opt.num("cw", 0.5));
    }
    sim.set_cw_linear(opt.num("cw-a", 0.0232), opt.num("cw-b", 0.00772));
    sim.set_out_every(opt.integer("out-every", 100));
    if (opt.has("qw-back-table")) {
        sim.set_back_table(opt.str("qw-back-table", ""));
    }
    sim.set_back_fuelgrad(opt.has("qw-back-fuelgrad"));


    sim.set_wind(wind_x, wind_y);


    if (coupling_step > 0) {
        ads::problems::fire_grasslands::coupling_config cc;
        cc.step = coupling_step;

        cc.kb = coupling_kb_override > 0 ? coupling_kb_override : 0.001563;
        cc.period = coupling_period;
        cc.max_cycles = coupling_max_cycles;
        cc.recenter = coupling_recenter;
        cc.search_radius_m = coupling_search_radius_m;

        if (opt.has("ns-legacy")) {

            cc.source = "temperature";
            cc.domain = "window";
            cc.time_mode = "fixed";
            cc.warm = false;
            cc.bc = "uniform";
            cc.normalize_sink = false;
        }
        cc.source = opt.str("ns-source", cc.source);
        if (cc.source != "rate" && cc.source != "temperature") {
            std::cerr << "--ns-source must be rate or temperature" << std::endl;
            return 1;
        }
        cc.domain = opt.str("ns-domain", cc.domain);
        cc.window_L_m = opt.num("ns-window-m", cc.window_L_m);
        cc.elem_m = opt.num("ns-elem-m", cc.elem_m);
        cc.margin_m = opt.num("ns-margin-m", cc.margin_m);
        cc.min_window_m = opt.num("ns-min-window-m", cc.min_window_m);
        cc.sample_spacing_m = opt.num("ns-sample-m", cc.sample_spacing_m);
        cc.N = opt.integer("ns-N", cc.N);
        cc.nsteps = opt.integer("ns-nsteps", cc.nsteps);
        cc.t_end = opt.num("ns-t-end", cc.t_end);
        cc.time_mode = opt.str("ns-time", cc.time_mode);
        if (opt.has("ns-nu")) {

            cc.Re = 1.0e6 / opt.num("ns-nu", 50.0);
        }
        cc.Re = opt.num("ns-Re", cc.Re);
        if (opt.has("ns-warm")) {
            cc.warm = true;
        }
        if (opt.has("ns-cold")) {
            cc.warm = false;
        }
        cc.bc = opt.str("ns-bc", cc.bc);
        cc.fields_n = opt.integer("ns-fields-n", cc.fields_n);
        if (opt.has("ns-normalize-sink")) {
            cc.normalize_sink = opt.integer("ns-normalize-sink", 1) != 0;
        }
        cc.save_fields = !opt.has("ns-no-fields");
        cc.fields_every = std::max(1, opt.integer("ns-fields-every", cc.fields_every));
        cc.keep_io = !opt.has("ns-no-keep-io");
        cc.factor_cache = opt.has("ns-factor-cache");
        cc.inproc = opt.has("ns-inproc");
        cc.ns_threads = opt.integer("ns-threads", cc.ns_threads);
        cc.dt_max_s = opt.num("ns-dt-max", cc.dt_max_s);
        cc.inflow_diag = opt.has("ns-inflow-diag");
        cc.rot_diag = opt.has("ns-rot-diag");
        {
            const std::string slv = opt.str("ns-solver", "kron");
            if (slv != "mumps" && slv != "kron") {
                std::cerr << "unknown --ns-solver=" << slv << " (mumps|kron)" << std::endl;
                return 2;
            }
            cc.kron = slv == "kron";
            cc.diag_every = std::max(1, opt.integer("ns-diag-every", 1));
        }
        cc.csv_every = std::max(1, opt.integer("ns-csv-every", cc.csv_every));
        cc.sink_law = opt.str("ns-sink-law", cc.sink_law);
        cc.kb_phys = opt.num("ns-kb-phys", cc.kb_phys);
        if (cc.sink_law != "legacy" && cc.sink_law != "rate" && cc.sink_law != "fuel") {
            std::cerr << "--ns-sink-law must be legacy, rate or fuel" << std::endl;
            return 2;
        }
        if (cc.sink_law == "fuel" && cc.source != "rate") {
            std::cerr << "--ns-sink-law=fuel needs --ns-source=rate" << std::endl;
            return 2;
        }
        if (cc.inproc && cc.domain != "full") {
            std::cerr << "--ns-inproc works only with --ns-domain=full (got '" << cc.domain
                      << "'): the in-process wind solver keeps one fixed N-S mesh" << std::endl;
            return 1;
        }
        cc.timing = opt.has("ns-timing");
        sim.set_coupling(cc);
        std::cout << "  Coupling: step=" << cc.step << " period=" << cc.period
                  << " max_cycles=" << cc.max_cycles
                  << " recenter=" << (cc.recenter ? "on" : "off")
                  << " search_radius_m=" << cc.search_radius_m
                  << " kb=" << cc.kb << " source=" << cc.source << " Re=" << cc.Re << " domain=" << cc.domain
                  << " time=" << cc.time_mode << " t_end=" << cc.t_end << " nsteps=" << cc.nsteps
                  << " warm=" << (cc.warm ? "on" : "off") << " bc=" << cc.bc << std::endl;
        if (cc.sink_law != "legacy") {  // v6 (2b); legacy prints nothing extra (log unchanged)
            std::cout << "  Sink law: " << cc.sink_law << " kb_phys="
                      << (cc.kb_phys > 0 ? cc.kb_phys
                                         : cc.kb * fire_time_s / (cc.sink_law == "fuel" ? 300.0 : 1.0))
                      << " (kb_ns=kb_phys/t_u)" << std::endl;
        }
    }

    sim.run();

    std::cout << "Simulation completed." << std::endl;
    std::cout << "Output files: out_*.data, fuel_*.data" << std::endl;
}
