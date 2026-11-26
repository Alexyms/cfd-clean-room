#include "cfdcore/flow.hpp"

#include "cfdcore/positions.hpp"
#include "cfdcore/scalar.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cfdcore::flow {

namespace {

using lanes::mask;

void require(bool condition, const char* what) {
    if (!condition) {
        throw std::invalid_argument(what);
    }
}

bool positive_and_finite(double x) { return std::isfinite(x) && x > 0.0; }

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// The operands of a cell-centre block: a row's low faces and its high faces, the
// next faces along the component's direction.
struct centre_operands {
    const double* low;
    const double* high;
};

// N cells' centre values, the reference's `0.5 * (a + b)`. The row's pointers are
// read once before the loop: read through the struct in it, GCC reloads them per
// cell and keeps the loop scalar.
template <std::size_t N>
CFDCORE_INLINE void centre_cells(lanes::cells<N>, centre_operands in, std::size_t m,
                                 double* CFDCORE_RESTRICT out) {
    const double* low = in.low + m;
    const double* high = in.high + m;
    double* row = out + m;
    for (std::size_t i = 0; i < N; ++i) {
        row[i] = 0.5 * (low[i] + high[i]);
    }
}

// The operands of a change's block.
struct change_operands {
    const double* now;
    const double* before;
    const mask* cells;
};

// N cells' absolute changes where the mask is set, +0.0 elsewhere, into each lane's
// running maximum, NaN kept.
template <std::size_t N>
CFDCORE_INLINE void change_cells(lanes::cells<N>, change_operands in, std::size_t m,
                                 double* CFDCORE_RESTRICT largest) {
    for (std::size_t i = 0; i < N; ++i) {
        const double change = lanes::keep(in.cells[m + i], std::fabs(in.now[m + i] - in.before[m + i]));
        largest[i] = scalar::np_maximum(change, largest[i]);
    }
}

// The operands of the effective viscosity's block.
struct viscosity_operands {
    const double* nu_t;
    const mask* live;
    double mu;
    double rho;
};

// N cells of `mu + rho * np.where(live, nu_t, 0.0)`.
template <std::size_t N>
void effective_cells(lanes::cells<N>, viscosity_operands in, std::size_t m,
                     double* CFDCORE_RESTRICT out) {
    for (std::size_t i = m; i < m + N; ++i) {
        out[i] = in.mu + in.rho * lanes::select(in.live[i], in.nu_t[i], 0.0);
    }
}

// The operands of the eddy viscosity's checks.
struct check_operands {
    const double* nu_t;
    const mask* live;
};

// N cells' flags of a live value that is not finite, and of one that is negative.
template <std::size_t N>
CFDCORE_INLINE void check_cells(lanes::cells<N>, check_operands in, std::size_t m,
                                mask& not_finite, mask& negative) {
    for (std::size_t i = m; i < m + N; ++i) {
        const double x = in.nu_t[i];
        // `x - x` is NaN for an infinity or a NaN and zero otherwise.
        not_finite |= in.live[i] & lanes::mask_of((x - x) != 0.0);
        negative |= in.live[i] & lanes::mask_of(x < 0.0);
    }
}

}  // namespace

double python_max(std::span<const double> values) {
    require(!values.empty(), "flow: the max of no value");
    double m = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > m) {
            m = values[i];
        }
    }
    return m;
}

template <int D>
void cell_centres(const layout::layout<D>& fields, int c, std::span<const double> velocity,
                  std::span<double> out) {
    require(c >= 0 && c < D, "flow: no such component");
    require(out.size() == fields.cell_count(), "flow: a cell field is not one value per cell");
    require(velocity.size() == fields.face_count(c), "flow: a velocity is not one value per face");
    const std::size_t n = fields.cells(0);
    const std::size_t stride = fields.face_stride(c);
    for (std::size_t s = 0; s < fields.cell_count(); s += n) {
        const double* low = velocity.data() + fields.low_face(c, s) - s;
        const centre_operands in{low, low + stride};
        lanes::for_each_block(s, s + n, [&](auto cells, std::size_t m) {
            centre_cells(cells, in, m, out.data());
        });
    }
}

double largest_change(std::span<const double> now, std::span<const double> before,
                      std::span<const lanes::mask> cells) {
    require(now.size() == before.size() && cells.size() == now.size(),
            "flow: the fields of a change differ in length");
    // A maximum per lane of a block, combined after the walk.
    double largest[lanes::width];
    std::fill(largest, largest + lanes::width, 0.0);
    const change_operands in{now.data(), before.data(), cells.data()};
    lanes::for_each_block(0, now.size(), [&](auto block, std::size_t m) {
        change_cells(block, in, m, largest);
    });
    double worst = largest[0];
    for (std::size_t i = 1; i < lanes::width; ++i) {
        worst = scalar::np_maximum(largest[i], worst);
    }
    return worst;
}

bool effective_viscosity(std::span<const double> nu_t, std::span<const lanes::mask> live,
                         double mu, double rho, std::span<double> out, bool& not_finite) {
    require(live.size() == nu_t.size() && out.size() == nu_t.size(),
            "flow: the fields of an effective viscosity differ in length");
    mask bad = lanes::none;
    mask negative = lanes::none;
    const check_operands checked{nu_t.data(), live.data()};
    lanes::for_each_block(0, nu_t.size(), [&](auto cells, std::size_t m) {
        check_cells(cells, checked, m, bad, negative);
    });
    not_finite = bad != lanes::none;
    if (not_finite || negative != lanes::none) {
        return false;
    }
    const viscosity_operands in{nu_t.data(), live.data(), mu, rho};
    lanes::for_each_block(0, out.size(), [&](auto cells, std::size_t m) {
        effective_cells(cells, in, m, out.data());
    });
    return true;
}

template <int D>
    requires(D >= 1)
std::string_view solver<D>::check(const grid::grid<D>& g, const modules<D>& m,
                                  const flow_input<D>& input) {
    if (m.predictor == nullptr || m.corrector == nullptr || m.outlets == nullptr) {
        return "the predictor, the corrector and the outlets are required";
    }
    if (&m.predictor->mesh() != &g || &m.corrector->mesh() != &g) {
        return "a module is on another grid";
    }
    const bool coupled = input.mode == viscosity_mode::coupled;
    if (coupled != (m.model != nullptr) || coupled != (m.walls != nullptr)) {
        return "the turbulence model and its boundary are given exactly when coupled";
    }
    if (coupled && (&m.model->mesh() != &g || &m.walls->mesh() != &g)) {
        return "a module is on another grid";
    }
    const std::size_t n = g.fields().cell_count();
    if ((input.mode == viscosity_mode::fixed) != !input.mu_eff.empty() ||
        (!input.mu_eff.empty() && input.mu_eff.size() != n)) {
        return "a fixed viscosity is one value per cell, given exactly when fixed";
    }
    if (!positive_and_finite(input.mu) || !positive_and_finite(input.rho) ||
        !positive_and_finite(input.nu) || !positive_and_finite(input.residual_scale)) {
        return "a viscosity, the density or the residual's scale is not finite and positive";
    }
    // The field is fixed for the solve, so it is read once, as the coupled path
    // reads its own each iteration before `predict` does (`_check_mu_eff`).
    if (input.mode == viscosity_mode::fixed) {
        switch (m.predictor->check_viscosity(input.mu_eff)) {
            case momentum::viscosity_status::not_finite:
                return "mu_eff must be finite in every non-SOLID cell";
            case momentum::viscosity_status::not_positive:
                return "mu_eff must be positive in every non-SOLID cell";
            case momentum::viscosity_status::ok:
                break;
        }
    }
    if (coupled && !(input.alpha_turbulence > 0.0 && input.alpha_turbulence <= 1.0)) {
        return "the eddy viscosity's relaxation is outside (0, 1]";
    }
    if (input.rule == stop_rule::velocity_step) {
        if (!positive_and_finite(input.convergence_tol)) {
            return "the velocity step's bound is not finite and positive";
        }
    } else {
        const std::string_view reason = stopping::rule::check(input.keys);
        if (!reason.empty()) {
            return reason;
        }
        if (input.keys.viscosity_condition != coupled) {
            return "condition (e) is applied exactly when coupled";
        }
    }
    const auto types = g.cell_types();
    if (std::none_of(types.begin(), types.end(),
                     [&](std::int32_t t) { return t == input.fluid; })) {
        return "no cell is FLUID";
    }
    return {};
}

template <int D>
    requires(D >= 1)
solver<D>::solver(const grid::grid<D>& g, const modules<D>& m, const flow_input<D>& input)
    : grid_(&g),
      modules_(m),
      mode_(input.mode),
      stop_rule_(input.rule),
      mu_(input.mu),
      rho_(input.rho),
      nu_(input.nu),
      alpha_turbulence_(input.alpha_turbulence),
      residual_scale_(input.residual_scale),
      convergence_tol_(input.convergence_tol) {
    const std::string_view reason = check(g, m, input);
    if (!reason.empty()) {
        throw std::invalid_argument(std::string("flow: ") + std::string(reason));
    }
    if (input.rule == stop_rule::error_estimate) {
        rule_.emplace(input.keys);
    }
    const layout::layout<D>& fields = g.fields();
    const std::size_t n = fields.cell_count();
    fluid_.assign(n, lanes::none);
    live_.assign(n, lanes::none);
    for (std::size_t i = 0; i < n; ++i) {
        fluid_[i] = lanes::mask_of(g.cell_types()[i] == input.fluid);
        live_[i] = lanes::mask_of(!g.solid(i));
        if (fluid_[i] != lanes::none) {
            fluid_cells_.push_back(i);
        }
    }
    fixed_mu_.assign(input.mu_eff.begin(), input.mu_eff.end());
    for (int c = 0; c < D; ++c) {
        for (auto* field : {&velocity_[c], &star_[c], &diagonal_[c], &corrected_[c]}) {
            field->assign(fields.face_count(c), 0.0);
        }
        centres_[c].assign(n, 0.0);
        previous_[c].assign(n, 0.0);
    }
    for (auto* field : {&p_, &p_next_, &p_prime_, &imbalance_, &magnitude_}) {
        field->assign(n, 0.0);
    }
    if (coupled()) {
        for (auto* field : {&k_, &eps_, &nu_t_, &mu_eff_, &k_next_, &eps_next_, &nu_t_step_,
                            &nu_t_next_}) {
            field->assign(n, 0.0);
        }
        median_.assign(fluid_cells_.size(), 0.0);
        for (int c = 0; c < D; ++c) {
            for (int q = 0; q < D - 1; ++q) {
                const std::size_t corners =
                    positions::faces_layout(fields, c, momentum::transverse(c, q)).cell_count();
                stencil_mu_[c][q].assign(corners, 0.0);
                wall_mu_[c][q].assign(corners, 0.0);
            }
        }
    }
}

template <int D>
    requires(D >= 1)
void solver<D>::start(const std::array<std::span<const double>, D>& velocity,
                      std::span<const double> p, std::span<const double> k,
                      std::span<const double> eps, std::span<const double> nu_t) {
    const layout::layout<D>& fields = grid_->fields();
    for (int c = 0; c < D; ++c) {
        require(velocity[c].size() == velocity_[c].size(),
                "flow: a velocity is not one value per face");
    }
    require(p.size() == p_.size(), "flow: p is not one value per cell");
    const std::size_t expected = coupled() ? p_.size() : 0;
    require(k.size() == expected && eps.size() == expected && nu_t.size() == expected,
            "flow: k, eps and nu_t are one value per cell when coupled and empty otherwise");
    for (int c = 0; c < D; ++c) {
        std::copy(velocity[c].begin(), velocity[c].end(), velocity_[c].begin());
        cell_centres<D>(fields, c, velocity_[c], centres_[c]);
    }
    std::copy(p.begin(), p.end(), p_.begin());
    std::copy(k.begin(), k.end(), k_.begin());
    std::copy(eps.begin(), eps.end(), eps_.begin());
    std::copy(nu_t.begin(), nu_t.end(), nu_t_.begin());
    if (rule_) {
        rule_->reset();
    }
}

template <int D>
    requires(D >= 1)
stopping::restore_status solver<D>::restore(std::span<const double> steps,
                                            std::span<const double> viscosity_steps) {
    if (!rule_) {
        return steps.empty() && viscosity_steps.empty() ? stopping::restore_status::ok
                                                        : stopping::restore_status::no_rule;
    }
    return rule_->restore(steps, viscosity_steps);
}

template <int D>
    requires(D >= 1)
std::array<std::span<const double>, D> solver<D>::const_velocity() const {
    std::array<std::span<const double>, D> out;
    for (int c = 0; c < D; ++c) {
        out[c] = velocity_[c];
    }
    return out;
}

template <int D>
    requires(D >= 1)
bool solver<D>::predict(chunk_failure& failure) {
    std::array<std::span<double>, D> star;
    std::array<std::span<double>, D> diagonal;
    for (int c = 0; c < D; ++c) {
        star[c] = star_[c];
        diagonal[c] = diagonal_[c];
    }
    const momentum::per_pair<std::span<const double>, D> no_walls{};
    momentum::predictor<D>& predictor = *modules_.predictor;
    if (mode_ == viscosity_mode::laminar) {
        predictor.predict(const_velocity(), p_, {}, no_walls, star, diagonal);
        return true;
    }
    if (mode_ == viscosity_mode::fixed) {
        predictor.predict(const_velocity(), p_, fixed_mu_, no_walls, star, diagonal);
        return true;
    }
    // `_effective_viscosity(state.nu_t)`: its two refusals, then its arithmetic.
    bool not_finite = false;
    if (!effective_viscosity(nu_t_, live_, mu_, rho_, mu_eff_, not_finite)) {
        failure.kind =
            not_finite ? failure::eddy_viscosity_not_finite : failure::eddy_viscosity_negative;
        return false;
    }
    // `stencil_viscosity` checks the field first, as `predict` does again.
    switch (predictor.check_viscosity(mu_eff_)) {
        case momentum::viscosity_status::not_finite:
            failure.kind = failure::mu_eff_not_finite;
            return false;
        case momentum::viscosity_status::not_positive:
            failure.kind = failure::mu_eff_not_positive;
            return false;
        case momentum::viscosity_status::ok:
            break;
    }
    momentum::per_pair<std::span<double>, D> stencil{};
    momentum::per_pair<std::span<const double>, D> elsewhere{};
    momentum::per_pair<std::span<double>, D> walls{};
    momentum::per_pair<std::span<const double>, D> wall_mu{};
    for (int c = 0; c < D; ++c) {
        for (int q = 0; q < D - 1; ++q) {
            stencil[c][q] = stencil_mu_[c][q];
            elsewhere[c][q] = stencil_mu_[c][q];
            walls[c][q] = wall_mu_[c][q];
            wall_mu[c][q] = wall_mu_[c][q];
        }
    }
    predictor.stencil_viscosity(mu_eff_, stencil);
    modules_.walls->wall_viscosity(k_, elsewhere, walls);
    for (int c = 0; c < D; ++c) {
        for (int q = 0; q < D - 1; ++q) {
            switch (predictor.check_wall_viscosity(c, q, wall_mu[c][q])) {
                case momentum::viscosity_status::not_finite:
                    failure.kind = failure::wall_mu_not_finite;
                    failure.component = c;
                    return false;
                case momentum::viscosity_status::not_positive:
                    failure.kind = failure::wall_mu_not_positive;
                    failure.component = c;
                    return false;
                case momentum::viscosity_status::ok:
                    break;
            }
        }
    }
    predictor.predict(const_velocity(), p_, mu_eff_, wall_mu, star, diagonal);
    return true;
}

template <int D>
    requires(D >= 1)
bool solver<D>::correct(iteration_record& record, chunk_failure& failure) {
    std::array<std::span<const double>, D> star;
    std::array<std::span<const double>, D> diagonal;
    std::array<std::span<double>, D> corrected;
    for (int c = 0; c < D; ++c) {
        star[c] = star_[c];
        diagonal[c] = diagonal_[c];
        corrected[c] = corrected_[c];
    }
    const pressure::solve_result result =
        modules_.corrector->correct(star, diagonal, p_, corrected, p_next_, p_prime_);
    if (result.status != pressure::solve_status::ok) {
        failure.kind = failure::pressure;
        failure.pressure = result.status;
        return false;
    }
    record.pressure_iterations = result.iterations;
    record.pressure_products = result.products;
    record.reached_cap = result.reached_cap;
    for (int c = 0; c < D; ++c) {
        std::swap(velocity_[c], corrected_[c]);
    }
    std::swap(p_, p_next_);
    return true;
}

template <int D>
    requires(D >= 1)
bool solver<D>::turbulence_step(iteration_record& record, chunk_failure& failure) {
    const std::array<std::span<const double>, D> velocity = const_velocity();
    // `_check_faces`: the state was checked when it was formed, by the step before
    // or by the solve's set-up, and `_effective_viscosity` read its nu_t above.
    for (int c = 0; c < D; ++c) {
        if (!turbulence::all_finite(velocity[c])) {
            failure.kind = failure::faces_not_finite;
            return false;
        }
    }
    const turbulence::conditions<D> conditions =
        modules_.walls->conditions(velocity, k_, eps_);
    const turbulence::condition_check checked = modules_.model->check_formed(conditions);
    if (checked.status != turbulence::condition_status::ok) {
        failure.kind = failure::conditions;
        failure.conditions = checked;
        return false;
    }
    const turbulence::step_result result = modules_.model->step(
        {k_, eps_, nu_t_}, velocity, conditions, k_next_, eps_next_, nu_t_step_);
    // The reference logs its two warnings before it tests the signs, so a step that
    // fails the test hands them back too; a step that moved nothing ran no solve.
    record.k_fallbacks = result.solves[0].fallbacks;
    record.eps_fallbacks = result.solves[1].fallbacks;
    record.solves_converged = result.solves[0].converged && result.solves[1].converged;
    switch (result.status) {
        case turbulence::step_status::nothing_moves:
            failure.kind = failure::nothing_moves;
            return false;
        case turbulence::step_status::k_not_positive:
            failure.kind = failure::k_not_positive;
            failure.bad = result.bad;
            return false;
        case turbulence::step_status::eps_not_positive:
            failure.kind = failure::eps_not_positive;
            failure.bad = result.bad;
            return false;
        case turbulence::step_status::ok:
            break;
    }
    turbulence::relax(nu_t_, nu_t_step_, alpha_turbulence_, nu_t_next_);
    record.viscosity_step = largest_change(nu_t_next_, nu_t_, live_);
    std::swap(k_, k_next_);
    std::swap(eps_, eps_next_);
    std::swap(nu_t_, nu_t_next_);
    record.rule.nu_scale = stopping::viscosity_scale(nu_, nu_t_, fluid_cells_, median_);
    return true;
}

template <int D>
    requires(D >= 1)
void solver<D>::residual(iteration_record& record) {
    const layout::layout<D>& fields = grid_->fields();
    std::array<double, D> change{};
    for (int c = 0; c < D; ++c) {
        std::swap(previous_[c], centres_[c]);
        cell_centres<D>(fields, c, velocity_[c], centres_[c]);
        change[c] = largest_change(centres_[c], previous_[c], fluid_);
    }
    record.step = python_max(change);
    record.residual = record.step / residual_scale_;
}

template <int D>
    requires(D >= 1)
bool solver<D>::stop(iteration_record& record, chunk_failure& failure) {
    if (!rule_) {
        // A capped correction makes the step small without balancing the faces, so
        // the velocity step does not stop on one (ADR-013 B).
        return record.residual < convergence_tol_ && !record.reached_cap;
    }
    stopping::rule& r = *rule_;
    const double nu_scale = record.rule.nu_scale;
    // `update` refuses a scale before it records anything.
    if (coupled() && !positive_and_finite(nu_scale)) {
        failure.kind = failure::nu_scale;
        failure.nu_scale = nu_scale;
        return false;
    }
    const bool estimates = r.record(record.step, record.viscosity_step, nu_scale);
    record.rule = r.last();
    if (!estimates) {
        return false;
    }
    std::array<std::span<const double>, D> velocity = const_velocity();
    modules_.corrector->mass_imbalance(velocity, imbalance_);
    return r.accept(stopping::summarise(imbalance_, magnitude_));
}

template <int D>
    requires(D >= 1)
chunk_result solver<D>::run(std::int64_t first, std::span<iteration_record> records) {
    using clock = std::chrono::steady_clock;
    chunk_result result;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t n = 0; n < records.size(); ++n) {
        iteration_record& record = records[n];
        record = iteration_record{};
        record.viscosity_step = nan;
        record.rule = {nan, nan, nan};
        chunk_failure& failure = result.failure;
        failure.iteration = first + static_cast<std::int64_t>(n);

        auto t0 = clock::now();
        modules_.outlets->extrapolate([&] {
            std::array<std::span<double>, D> faces;
            for (int c = 0; c < D; ++c) {
                faces[c] = velocity_[c];
            }
            return faces;
        }());
        const bool predicted = predict(failure);
        result.seconds.momentum += seconds_since(t0);
        if (!predicted) {
            return result;
        }

        t0 = clock::now();
        const bool corrected = correct(record, failure);
        result.seconds.pressure += seconds_since(t0);
        if (!corrected) {
            return result;
        }

        if (coupled()) {
            t0 = clock::now();
            const bool stepped = turbulence_step(record, failure);
            result.seconds.turbulence += seconds_since(t0);
            if (!stepped) {
                return result;
            }
        }

        t0 = clock::now();
        residual(record);
        result.seconds.correct += seconds_since(t0);

        t0 = clock::now();
        const bool stopped = stop(record, failure);
        result.seconds.stopping += seconds_since(t0);
        if (failure.kind != failure::none) {
            return result;
        }
        result.outer = static_cast<std::int64_t>(n) + 1;
        if (stopped) {
            result.stopped = true;
            return result;
        }
    }
    return result;
}

template void cell_centres<2>(const layout::layout<2>&, int, std::span<const double>,
                              std::span<double>);
template class solver<2>;

}  // namespace cfdcore::flow
