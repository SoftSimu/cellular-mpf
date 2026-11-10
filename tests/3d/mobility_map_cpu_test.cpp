/* Cellular MPF; SPDX-License-Identifier: LGPL-3.0-or-later */
#include "pf3d/mobility_map.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const fs::path directory = fs::temp_directory_path() / ("pf3d-mobility-" + std::to_string(stamp));
    if (!fs::create_directory(directory)) return 1;
    const fs::path path = directory / "map.txt";
    bool passed = true;
    auto check = [&](const std::string& contents, bool expected) {
        { std::ofstream output(path); output << contents; }
        std::map<std::int64_t, float> values;
        std::string error;
        const bool result = pf3d::read_mobility_map(path.string(), &values, &error);
        if (result != expected || (!result && error.empty())) passed = false;
        return values;
    };
    const auto good = check("# cell_id absolute_M\n2 0.175 # reduced\n7 29.4\n", true);
    passed = passed && good.at(2) == 0.175f && good.at(7) == 29.4f;
    for (const auto* invalid : {"2 0", "2 -1", "2 nan", "2 inf", "2 1e100",
                               "2 1e-100", "-1 1", "1.5 1", "2 1 extra",
                               "2", "2 1\n2 2"}) check(invalid, false);
    check("\n# empty overlay\n", true);
    fs::remove(path);
    fs::remove(directory);
    if (!passed) std::cerr << "mobility-map parsing failed\n";
    return passed ? 0 : 1;
}
