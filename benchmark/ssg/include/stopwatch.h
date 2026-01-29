#pragma once

#include <chrono>

class StopW {
    std::chrono::steady_clock::time_point time_begin;

public:
    StopW() { time_begin = std::chrono::steady_clock::now(); }

    long long getElapsedTimeMicro() {
        auto time_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count();
    }

    void reset() { time_begin = std::chrono::steady_clock::now(); }
};

// Memory statistics are provided by efanna2e/util.h (getPeakRSS, getCurrentRSS)
