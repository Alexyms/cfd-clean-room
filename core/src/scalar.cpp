#include "cfdcore/scalar.hpp"

#include <cstddef>

namespace cfdcore::scalar {

namespace {

// One line of cells along a direction, with its two boundary nodes: node m is
// the low boundary node for m = 0, cell m - 1 for m = 1..n, and the high
// boundary node for m = n + 1, as the reference's `ext` holds them.
template <int D>
class line_view {
public:
    line_view(const grid::grid<D>& g, std::span<const double> c, const layout::line& line,
              std::size_t stride, std::size_t n, double low, double high)
        : g_(g), c_(c), cell_(line.cell), stride_(stride), n_(n), low_(low), high_(high) {}

    double node(std::size_t m) const {
        if (m == 0) {
            return low_;
        }
        if (m == n_ + 1) {
            return high_;
        }
        return c_[cell_ + (m - 1) * stride_];
    }

    // mesh_axes pads the SOLID mask with one False on each side.
    bool solid_node(std::size_t m) const {
        if (m == 0 || m == n_ + 1) {
            return false;
        }
        return g_.solid(cell_ + (m - 1) * stride_);
    }

private:
    const grid::grid<D>& g_;
    std::span<const double> c_;
    std::size_t cell_;
    std::size_t stride_;
    std::size_t n_;
    double low_;
    double high_;
};

// QUICK's face value at interior face f (between cells f and f + 1, nodes
// f + 1 and f + 2), as quick_face_values forms f_pos and f_neg: three
// products added left to right.
template <int D>
double quick_at(const quick_weights& w, std::size_t f, bool positive, const line_view<D>& line) {
    if (positive) {
        return w.positive[0][f] * line.node(f) + w.positive[1][f] * line.node(f + 1) +
               w.positive[2][f] * line.node(f + 2);
    }
    return w.negative[0][f] * line.node(f + 3) + w.negative[1][f] * line.node(f + 2) +
           w.negative[2][f] * line.node(f + 1);
}

}  // namespace

void limited_face_values(std::span<const double> c_up, std::span<const double> c_c,
                         std::span<const double> c_d, std::span<const double> quick,
                         std::span<double> out) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = limited_face_value(c_up[i], c_c[i], c_d[i], quick[i]);
    }
}

template <int D>
    requires(D >= 1)
scheme<D>::scheme(const grid::grid<D>& g) : grid_(&g) {
    for (int d = 0; d < D; ++d) {
        const auto nodes = g.nodes(d);
        const auto faces = g.interior_faces(d);
        auto& w = weights_[d];
        for (int i = 0; i < 3; ++i) {
            w.positive[i].resize(faces.size());
            w.negative[i].resize(faces.size());
        }
        // quick_face_values with left k = f + 1: far_pos = k - 1 and
        // far_neg = k + 2, both inside the nodes, which run from the low
        // boundary face to the high one.
        for (std::size_t f = 0; f < faces.size(); ++f) {
            const auto pos = lagrange_weights(nodes[f], nodes[f + 1], nodes[f + 2], faces[f]);
            const auto neg = lagrange_weights(nodes[f + 3], nodes[f + 2], nodes[f + 1], faces[f]);
            for (int i = 0; i < 3; ++i) {
                w.positive[i][f] = pos[i];
                w.negative[i][f] = neg[i];
            }
        }
    }
}

template <int D>
void quick_face_values(const scheme<D>& s, int d, std::span<const double> c,
                       std::span<const double> c_low, std::span<const double> c_high,
                       std::span<const bool> positive, std::span<double> out) {
    const auto& g = s.mesh();
    const auto& fields = g.fields();
    const std::size_t n = fields.cells(d);
    const std::size_t stride = fields.cell_stride(d);
    const auto& w = s.weights(d);
    const auto lines = fields.lines(d);
    for (std::size_t l = 0; l < lines.size(); ++l) {
        const line_view<D> line(g, c, lines[l], stride, n, c_low[l], c_high[l]);
        for (std::size_t f = 0; f + 1 < n; ++f) {
            const std::size_t face = lines[l].face + (f + 1) * stride;
            out[face] = quick_at(w, f, positive[face], line);
        }
    }
}

template <int D>
void advective_flux(const scheme<D>& s, int d, std::span<const double> c,
                    std::span<const double> flux, std::span<const double> inflow, bool upwind,
                    std::span<const bool> vents, std::span<double> out) {
    const auto& g = s.mesh();
    const auto& fields = g.fields();
    const std::size_t n = fields.cells(d);
    const std::size_t stride = fields.cell_stride(d);
    const auto& w = s.weights(d);
    const bool with_vents = !vents.empty();
    for (const auto& ln : fields.lines(d)) {
        const std::size_t low_face = ln.face;
        const std::size_t high_face = ln.face + n * stride;
        const double c_low = flux[low_face] > 0.0 ? inflow[low_face] : c[ln.cell];
        const double c_high =
            flux[high_face] < 0.0 ? inflow[high_face] : c[ln.cell + (n - 1) * stride];
        const line_view<D> line(g, c, ln, stride, n, c_low, c_high);
        out[low_face] = flux[low_face] * c_low;
        // Interior face f lies between nodes k = f + 1 and k + 1, cells f and f + 1.
        for (std::size_t f = 0; f + 1 < n; ++f) {
            const std::size_t face = ln.face + (f + 1) * stride;
            const std::size_t k = f + 1;
            const double interior = flux[face];
            const bool positive = interior > 0.0;
            const double c_c = positive ? line.node(k) : line.node(k + 1);
            double value = c_c;
            if (!upwind) {
                const double c_d = positive ? line.node(k + 1) : line.node(k);
                double c_up = positive ? line.node(k - 1) : line.node(k + 2);
                double quick = quick_at(w, f, positive, line);
                // A far node inside an obstacle is read as the upstream value,
                // the zero-gradient rule a domain wall gets.
                const bool far_solid = positive ? line.solid_node(k - 1) : line.solid_node(k + 2);
                if (far_solid) {
                    c_up = c_c;
                    quick = c_c;
                }
                value = limited_face_value(c_up, c_c, c_d, quick);
            }
            if (with_vents && vents[face]) {
                const bool solid_low = line.solid_node(k);
                const bool solid_high = line.solid_node(k + 1);
                const bool entering =
                    (solid_low && interior > 0.0) || (solid_high && interior < 0.0);
                const double room = solid_low ? line.node(k + 1) : line.node(k);
                value = entering ? inflow[face] : room;
            }
            out[face] = interior * value;
        }
        out[high_face] = flux[high_face] * c_high;
    }
}

template class scheme<1>;
template class scheme<2>;

template void quick_face_values<1>(const scheme<1>&, int, std::span<const double>,
                                   std::span<const double>, std::span<const double>,
                                   std::span<const bool>, std::span<double>);
template void quick_face_values<2>(const scheme<2>&, int, std::span<const double>,
                                   std::span<const double>, std::span<const double>,
                                   std::span<const bool>, std::span<double>);
template void advective_flux<1>(const scheme<1>&, int, std::span<const double>,
                                std::span<const double>, std::span<const double>, bool,
                                std::span<const bool>, std::span<double>);
template void advective_flux<2>(const scheme<2>&, int, std::span<const double>,
                                std::span<const double>, std::span<const double>, bool,
                                std::span<const bool>, std::span<double>);

}  // namespace cfdcore::scalar
