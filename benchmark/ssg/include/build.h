#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dataset.h"
#include "file_io.h"
#include "index_ssg.h"
#include "logging.h"
#include "parameters.h"
#include "stopwatch.h"
#include "util.h"


namespace ssg::benchmark {

struct CreateGraphParams {
    // NNDescent parameters
    unsigned K = 50;     // Number of neighbors
    unsigned L = 70;     // Candidate list size
    unsigned iter = 10;  // NNDescent iterations
    unsigned S = 10;     // Sampling parameter
    unsigned R = 50;     // RNG parameter

    // KDTree parameters (optional, set nTrees > 0 to enable)
    unsigned nTrees = 0;  // Number of trees (0 = no KDTree)
    unsigned mLevel = 8;  // Max level for KDTree

    // Test parameters
    uint32_t anns_k = 100;
    uint32_t anns_repeat = 1;
    uint32_t explore_k = 1000;
    uint32_t explore_repeat = 1;

    std::vector<unsigned> L_search = {100, 200, 300, 500};
};

inline void wait_before_test(int seconds = 5) {
    log("Waiting %d seconds for machine to settle...\n", seconds);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

// Load ivecs file as vector of vectors (for ground truth / entry nodes)
inline std::vector<std::vector<uint32_t>> load_ivecs_as_vectors(const char* filename, size_t& count) {
    size_t d = 0, n = 0;
    auto ptr = ivecs_read(filename, d, n);
    count = n;

    std::vector<std::vector<uint32_t>> res(n);
    if (!ptr) return res;

    for (size_t i = 0; i < n; ++i) {
        res[i].assign(ptr.get() + i * d, ptr.get() + (i + 1) * d);
        std::sort(res[i].begin(), res[i].end());
    }
    return res;
}

// SSG Parameters
struct SSGBuildParams {
    unsigned L = 100;
    unsigned R = 50;
    unsigned A = 60;
};

inline efanna2e::Parameters to_ssg_params(const SSGBuildParams& ssg) {
    efanna2e::Parameters params;
    params.Set<unsigned>("L", ssg.L);
    params.Set<unsigned>("R", ssg.R);
    params.Set<unsigned>("A", ssg.A);
    params.Set<unsigned>("n_try", 10);
    return params;
}

// Build SSG index from existing KNN graph
inline std::unique_ptr<efanna2e::IndexSSG> build_ssg_index(
    const float* data, size_t n, size_t dim, const SSGBuildParams& params, const std::string& knn_graph_path) {
    auto index = std::make_unique<efanna2e::IndexSSG>(dim, n, efanna2e::L2, nullptr);

    // SSG Build requires a pre-built KNN graph
    efanna2e::Parameters ssg_params = to_ssg_params(params);
    ssg_params.Set<std::string>("nn_graph_path", knn_graph_path);

    log("Building SSG index...\n");
    StopW stopw;
    index->Build(n, data, ssg_params);
    log("SSG Build time: %.2f seconds. Mem: %zu Mb. Peak: %zu Mb.\n",
        1e-6 * stopw.getElapsedTimeMicro(),
        getCurrentRSS() / 1000000,
        getPeakRSS() / 1000000);

    return index;
}

// Load existing SSG index or build new one
inline std::unique_ptr<efanna2e::IndexSSG> load_or_build_ssg_index(const Dataset& ds,
                                                                   const float* data,
                                                                   size_t n,
                                                                   size_t dim,
                                                                   const SSGBuildParams& params,
                                                                   const std::string& knn_graph_path,
                                                                   const std::string& graph_path) {
    auto index = std::make_unique<efanna2e::IndexSSG>(dim, n, efanna2e::L2, nullptr);

    if (file_exists(graph_path)) {
        log("Loading existing SSG index from %s\n", graph_path.c_str());
        index->Load(graph_path.c_str());
        log("Mem: %zu Mb. Peak: %zu Mb.\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    } else {
        // Ensure KNN graph exists
        if (!file_exists(knn_graph_path)) {
            log("Error: KNN graph %s does not exist. Cannot build SSG.\n", knn_graph_path.c_str());
            return nullptr;
        }

        // Output directory is guaranteed by dataset logic usually, but let's be safe
        std::filesystem::path p(graph_path);
        ensure_directory(p.parent_path());

        index = build_ssg_index(data, n, dim, params, knn_graph_path);

        log("Saving SSG index to %s\n", graph_path.c_str());
        index->Save(graph_path.c_str());
    }

    return index;
}

}  // namespace ssg::benchmark
