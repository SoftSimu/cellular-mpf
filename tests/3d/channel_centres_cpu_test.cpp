#include "pf3d/palmieri_initializer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    namespace fs = std::filesystem;
    const auto suffix = std::chrono::high_resolution_clock::now()
                            .time_since_epoch().count();
    const fs::path directory = fs::temp_directory_path() /
        ("pf-channel-centres-" + std::to_string(suffix));
    if (!fs::create_directory(directory)) return 1;
    const fs::path path = directory / "centres.csv";
    int failures = 0;
    auto check = [&](const std::string& csv, bool expected,
                     bool geometry_pinned = false) {
        { std::ofstream file(path, std::ios::binary); file << csv; }
        std::vector<float> x, y, z;
        pf3d::PalmieriCentres3DCsvDiagnostics diagnostic{};
        std::string error;
        const bool valid = pf3d::channel3d_read_centres_csv(
            path.string(), 2, 200, 196, 49, &x, &y, &z,
            &diagnostic, &error);
        if (valid != expected ||
            (valid && diagnostic.channel_box_declared != geometry_pinned)) {
            std::cerr << "unexpected result: " << error << '\n' << csv;
            ++failures;
        }
        if (valid && (diagnostic.accepted_count != 2 ||
                      diagnostic.table_fnv1a64 == 0)) ++failures;
    };
    const std::string header = "global_id,x,y,z\n";
    const std::string box = "# channel_box=200,196\n";
    const std::string stack = "0,40,40,49\n1,130,110,147\n";
    check(header + "0,100,100,98\n1,10,10,49\n", true);
    check(header + stack, false);
    check(header + box + stack, true, true);
    check(header + "# channel_box=201,196\n" + stack, false);
    check(header + "# channel_box=200,198\n" + stack, false);
    check(header + "# channel_box=200,196,1\n" + stack, false);
    check(header + box + "0,40,40,48\n1,130,110,147\n", false);
    check(header + box + "0,40,40,49\n1,40,40,49\n", false);
    check(header + box + "0,40,40,49\n1,200,110,147\n", false);
    check(header + box + "0,40,40,49\n0,130,110,147\n", false);
    check(header + box + "0,nan,40,49\n1,130,110,147\n", false);
    check(header + box + "0,40,40,49\n", false);
    check(header + box + stack + "2,80,80,98\n", false);
    check(header + box + stack + "\n", false);
    check(header + box, false);
    check(header, false);
    fs::remove(path);
    fs::remove(directory);
    std::cout << "channel centre input: " << failures << " failures\n";
    return failures ? 1 : 0;
}
