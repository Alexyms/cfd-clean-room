#include "cfdcore/layout.hpp"

namespace cfdcore::layout {

template <int D>
    requires(D >= 1)
layout<D>::layout(std::array<std::size_t, D> cells) : cells_(cells) {
    std::size_t stride = 1;
    for (int d = 0; d < D; ++d) {
        cell_stride_[d] = stride;
        stride *= cells_[d];
    }
    cell_count_ = stride;
    for (int d = 0; d < D; ++d) {
        face_count_[d] = cell_count_ / cells_[d] * (cells_[d] + 1);
        // The cells at the low end of direction d, in ascending order, are
        // the lines' first cells with direction 0 the fastest among the rest.
        lines_[d].reserve(cell_count_ / cells_[d]);
        for (std::size_t cell = 0; cell < cell_count_; ++cell) {
            if ((cell / cell_stride_[d]) % cells_[d] == 0) {
                lines_[d].push_back(line{cell, low_face(d, cell)});
            }
        }
    }
}

template class layout<1>;
template class layout<2>;

}  // namespace cfdcore::layout
