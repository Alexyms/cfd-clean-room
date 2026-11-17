// The core's grid (ADR-017 C, module `grid`; decision 4).
//
// The geometry the explicit face flux reads, per direction, and the cell
// types: built once from the arrays `Mesh` holds, handed in through the
// binding, and read-only after. The mesh is built in Python (ADR-017 E); this
// is its copy in the core's own terms.

#pragma once

#include "cfdcore/layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace cfdcore::grid {

// What a grid is built from, per direction d, direction 0 first (x, then y).
//
// faces:     the n_d + 1 face coordinates along d, boundary faces included, m
//            (`Mesh.x`, `Mesh.y`)
// centres:   the n_d cell centre coordinates along d, m (`Mesh.xc`, `Mesh.yc`)
// cell_type: one type per cell, in the cell field's layout (layout.hpp):
//            `Mesh.cell_type` [ny, nx] for D = 2
// solid:     the type of a SOLID cell (`mesh.SOLID`), so the core holds no
//            copy of the mesh's codes
template <int D>
struct grid_input {
    std::array<std::span<const double>, D> faces;
    std::array<std::span<const double>, D> centres;
    std::span<const std::int32_t> cell_type;
    std::int32_t solid;
};

template <int D>
    requires(D >= 1)
class grid {
public:
    // Why `input` cannot make a grid, or an empty view when it can: a
    // direction without a cell, a faces array that is not one longer than
    // its centres, or a cell_type array that is not one per cell. The binding
    // turns a reason into the exception it raises.
    static std::string_view check(const grid_input<D>& input);

    // Builds the grid from an input `check` accepts, copying every array.
    explicit grid(const grid_input<D>& input);

    // The layout of the grid's cell and face fields.
    const layout::layout<D>& fields() const { return layout_; }

    // The cell count along direction d.
    std::size_t cells(int d) const { return layout_.cells(d); }

    // The n_d + 2 nodes along d, m: the low boundary face, the n_d cell
    // centres and the high boundary face (the reference's `Axis.nodes`).
    std::span<const double> nodes(int d) const { return nodes_[d]; }

    // The n_d - 1 interior face coordinates along d, m (`Axis.faces`).
    std::span<const double> interior_faces(int d) const { return interior_faces_[d]; }

    // The cell types, in the cell field's layout.
    std::span<const std::int32_t> cell_types() const { return cell_type_; }

    // Whether a cell is SOLID; cell indexes the cell field.
    bool solid(std::size_t cell) const { return solid_[cell] != 0; }

private:
    layout::layout<D> layout_;
    std::array<std::vector<double>, D> nodes_;
    std::array<std::vector<double>, D> interior_faces_;
    std::vector<std::int32_t> cell_type_;
    std::vector<std::uint8_t> solid_;
};

extern template class grid<1>;
extern template class grid<2>;

}  // namespace cfdcore::grid
