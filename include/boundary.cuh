/*
 * Cellular MPF
 *
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * See LICENSE.txt and COPYING for the license terms.
 */

#pragma once
#include "kernels.cuh"
#include <memory>
#include <string>

namespace pf {
class BoundaryOutput {
public:
    BoundaryOutput();
    ~BoundaryOutput();
    BoundaryOutput(const BoundaryOutput&) = delete;
    BoundaryOutput& operator=(const BoundaryOutput&) = delete;
    bool open(const std::string& path, const SimParams& p, int side,
              long long interval, bool compress);
    bool capture(const float* phi, const CellState* cells, cudaStream_t stream,
                 long long step, double time);
    bool close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace pf
