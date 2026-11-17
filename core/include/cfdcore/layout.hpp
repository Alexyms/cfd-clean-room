// The layout of the core's fields (ADR-017 C, module `layout`; decision 4).
//
// A cell field holds one value per cell, row-major with direction 0 the
// fastest: [ny, nx] for D = 2, as `Mesh` and the reference hold it. The face
// field normal to direction d holds one value per face normal to d, in the
// same order with n_d + 1 entries along d: [ny, nx + 1] for d = 0 (the
// reference's u faces) and [ny + 1, nx] for d = 1 (its v faces). The core
// never transposes a field: it walks it along a direction with that
// direction's stride (prompt 87, section 1).

#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace cfdcore::layout {

// One line of cells along a direction: the index of its first cell in the
// cell field and of its low boundary face in the face field normal to that
// direction. Cell m of the line is `cell + m * cell_stride(d)` and face m is
// `face + m * face_stride(d)`, m counting from the low end.
struct line {
    std::size_t cell;
    std::size_t face;
};

template <int D>
    requires(D >= 1)
class layout {
public:
    // cells: the cell count per direction, direction 0 first, each at least 1.
    explicit layout(std::array<std::size_t, D> cells);

    // The cell count along direction d.
    std::size_t cells(int d) const { return cells_[d]; }

    // The number of cells: the length of a cell field.
    std::size_t cell_count() const { return cell_count_; }

    // The offset from a cell to its neighbour on the high side in direction
    // d, in the cell field; the low neighbour is at minus this.
    std::size_t cell_stride(int d) const { return cell_stride_[d]; }

    // The number of faces normal to direction d: the length of that face field.
    std::size_t face_count(int d) const { return face_count_[d]; }

    // The offset between two faces normal to d that are adjacent along d, in
    // that face field. It equals cell_stride(d): the directions below d have
    // the same extents in both fields.
    std::size_t face_stride(int d) const { return cell_stride_[d]; }

    // The face normal to d on the low side of a cell, as an index into the
    // face field normal to d; the high face is face_stride(d) beyond it.
    // cell: an index into the cell field, below cell_count().
    std::size_t low_face(int d, std::size_t cell) const {
        return cell + (cell / (cell_stride_[d] * cells_[d])) * cell_stride_[d];
    }

    // The lines of cells along direction d, in the order of the cell field's
    // remaining directions (direction 0 fastest): for D = 2 and d = 1, one
    // line per column, i = 0 to nx - 1, the order of the reference's
    // transposed y call. Built once at construction.
    std::span<const line> lines(int d) const { return lines_[d]; }

    // A zero cell field and a zero face field normal to d, for allocating a
    // field once when a run or an interval is set up.
    std::vector<double> cell_field() const { return std::vector<double>(cell_count_, 0.0); }
    std::vector<double> face_field(int d) const {
        return std::vector<double>(face_count_[d], 0.0);
    }

private:
    std::array<std::size_t, D> cells_;
    std::array<std::size_t, D> cell_stride_;
    std::array<std::size_t, D> face_count_;
    std::size_t cell_count_;
    std::array<std::vector<line>, D> lines_;
};

extern template class layout<1>;
extern template class layout<2>;

}  // namespace cfdcore::layout
