// Positions in a layout (ADR-017 C, beside module `layout`): what a module that
// reads one layout's row at another layout's place needs, a face's corner say.

#pragma once

#include "cfdcore/layout.hpp"

#include <array>
#include <cstddef>

namespace cfdcore::positions {

// A position in a layout, one entry per direction. A type of its own, so that a
// function deduces D from it.
template <int D>
struct position : std::array<std::size_t, D> {};

// The position of an element of a layout.
template <int D>
position<D> position_of(const layout::layout<D>& fields, std::size_t index) {
    position<D> at{};
    for (int d = 0; d < D; ++d) {
        at[d] = (index / fields.cell_stride(d)) % fields.cells(d);
    }
    return at;
}

// The index of a position in a layout.
template <int D>
std::size_t index_of(const layout::layout<D>& fields, const position<D>& at) {
    std::size_t index = 0;
    for (int d = 0; d < D; ++d) {
        index += at[d] * fields.cell_stride(d);
    }
    return index;
}

// The index of a position in an array over every direction of a layout but
// `without`, the others in their order: an entry of a domain edge normal to it.
template <int D>
std::size_t index_without(const layout::layout<D>& fields, const position<D>& at, int without) {
    std::size_t index = 0;
    std::size_t stride = 1;
    for (int d = 0; d < D; ++d) {
        if (d != without) {
            index += at[d] * stride;
            stride *= fields.cells(d);
        }
    }
    return index;
}

// A position with one entry moved up.
template <int D>
position<D> above(position<D> at, int d) {
    ++at[d];
    return at;
}

// A position with one entry moved down; the entry must be at least 1.
template <int D>
position<D> below(position<D> at, int d) {
    --at[d];
    return at;
}

// The layout of the faces normal to direction c as one entry per face, and that
// of a pair's corners, where the faces of two directions cross: one more than
// the cells along each of `c` and `t` (the same direction twice for faces).
template <int D>
layout::layout<D> faces_layout(const layout::layout<D>& cells, int c, int t) {
    std::array<std::size_t, D> n{};
    for (int d = 0; d < D; ++d) {
        n[d] = cells.cells(d) + (d == c || d == t ? 1 : 0);
    }
    return layout::layout<D>(n);
}

}  // namespace cfdcore::positions
