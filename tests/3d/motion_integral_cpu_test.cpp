#include "pf3d/motion_integral.hpp"

#include <cmath>
#include <cstdio>

int main() {
    pf3d::PlanarMotionIntegral whole{}, split{};
    constexpr double dt = 0.01;
    double active_x = 0.0, advective_x = 0.0;
    for (int step = 0; step < 1000; ++step) {
        const float speed = step < 200 ? 0.0f : 0.01f;
        const float px = step < 600 ? 1.0f : -1.0f;
        const float vx = speed * px - 0.004f;
        whole.advance(dt, speed, px, 0.0f, vx, 0.002f);
        active_x += dt * static_cast<double>(speed * px);
        advective_x += dt * static_cast<double>(vx);
        if (step == 499) split = whole;
        if (step >= 500) split.advance(dt, speed, px, 0.0f, vx, 0.002f);
    }
    if (whole.propulsion_x != active_x || whole.advection_x != advective_x ||
        whole.propulsion_y != 0.0 || whole.advection_y != split.advection_y ||
        whole.propulsion_x != split.propulsion_x || whole.advection_x != split.advection_x ||
        std::abs(whole.advection_y - 1000*dt*static_cast<double>(0.002f)) > 1e-14) {
        std::fprintf(stderr, "accepted-step planar integral mismatch\n");
        return 1;
    }
    auto trial = whole;
    trial.advance(dt, 0.01f, 1.0f, 0.0f, 0.03f, 0.0f);
    trial = whole;
    if (trial.propulsion_x != whole.propulsion_x || trial.advection_x != whole.advection_x)
        return 2;
    return 0;
}
