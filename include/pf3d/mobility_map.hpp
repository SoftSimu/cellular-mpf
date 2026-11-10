/*
 * Cellular MPF
 * Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace pf3d {

// Values are absolute Allen--Cahn mobilities, not multipliers of gamma.
inline bool read_mobility_map(const std::string& path,
                              std::map<std::int64_t, float>* values,
                              std::string* error) {
    std::ifstream input(path);
    if (!input) { *error = "cannot open mobility map: " + path; return false; }
    std::string line;
    unsigned number = 0;
    while (std::getline(input, line)) {
        ++number;
        line.erase(line.find('#') == std::string::npos ? line.size() : line.find('#'));
        std::istringstream row(line);
        row >> std::ws;
        if (row.eof()) continue;
        std::int64_t id;
        double value;
        std::string extra;
        if (!(row >> id >> value) || row >> extra || id < 0 ||
            !std::isfinite(value) || !(value > 0.0) ||
            !std::isfinite(static_cast<float>(value)) ||
            !(static_cast<float>(value) > 0.0f)) {
            *error = "invalid mobility map row " + std::to_string(number);
            return false;
        }
        if (!values->emplace(id, static_cast<float>(value)).second) {
            *error = "duplicate mobility map cell id " + std::to_string(id);
            return false;
        }
    }
    if (!input.eof()) { *error = "error reading mobility map"; return false; }
    return true;
}

}  // namespace pf3d
