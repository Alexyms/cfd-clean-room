#include "cfdcore/grid.hpp"

#include <cmath>

namespace cfdcore::grid {

namespace {

// Whether each value is below the next. A NaN anywhere gives false.
bool strictly_increasing(std::span<const double> values) {
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        if (!(values[i] < values[i + 1])) {
            return false;
        }
    }
    return true;
}

template <int D>
std::array<std::size_t, D> cell_counts(const grid_input<D>& input) {
    std::array<std::size_t, D> counts{};
    for (int d = 0; d < D; ++d) {
        counts[d] = input.centres[d].size();
    }
    return counts;
}

}  // namespace

template <int D>
    requires(D >= 1)
std::string_view grid<D>::check(const grid_input<D>& input) {
    std::size_t cells = 1;
    for (int d = 0; d < D; ++d) {
        if (input.centres[d].empty()) {
            return "a direction has no cell";
        }
        if (input.faces[d].size() != input.centres[d].size() + 1) {
            return "a direction's faces are not one more than its centres";
        }
        cells *= input.centres[d].size();
    }
    if (input.cell_type.size() != cells) {
        return "cell_type does not hold one type per cell";
    }
    return {};
}

template <int D>
    requires(D >= 1)
int grid<D>::first_bad_direction(const grid_input<D>& input) {
    for (int d = 0; d < D; ++d) {
        const auto faces = input.faces[d];
        const auto centres = input.centres[d];
        for (const double face : faces) {
            if (!std::isfinite(face)) {
                return d;
            }
        }
        if (!strictly_increasing(faces)) {
            return d;
        }
        // The nodes are faces.front(), the centres and faces.back().
        if (!(faces.front() < centres.front()) || !strictly_increasing(centres) ||
            !(centres.back() < faces.back())) {
            return d;
        }
    }
    return -1;
}

template <int D>
    requires(D >= 1)
grid<D>::grid(const grid_input<D>& input)
    : layout_(cell_counts(input)),
      cell_type_(input.cell_type.begin(), input.cell_type.end()),
      solid_(input.cell_type.size()) {
    for (int d = 0; d < D; ++d) {
        const auto faces = input.faces[d];
        const auto centres = input.centres[d];
        // As mesh_axes concatenates them: the low face, the centres, the high face.
        nodes_[d].reserve(centres.size() + 2);
        nodes_[d].push_back(faces.front());
        nodes_[d].insert(nodes_[d].end(), centres.begin(), centres.end());
        nodes_[d].push_back(faces.back());
        interior_faces_[d].assign(faces.begin() + 1, faces.end() - 1);
    }
    for (std::size_t cell = 0; cell < cell_type_.size(); ++cell) {
        solid_[cell] = cell_type_[cell] == input.solid ? 1 : 0;
    }
}

template class grid<1>;
template class grid<2>;

}  // namespace cfdcore::grid
