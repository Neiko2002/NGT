#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "NGT/Index.h"
#include "dataset.h"
#include "file_io.h"
#include "logging.h"
#include "stopwatch.h"
#include "util.h"

namespace ngt::benchmark {

struct CreateGraphParams {
    // NNDescent parameters (Not used in NGT directly, but maybe for consistency)
    unsigned K = 50;     // Number of neighbors
    unsigned L = 70;     // Candidate list size
    unsigned iter = 10;  // NNDescent iterations
    unsigned S = 10;     // Sampling parameter
    unsigned R = 50;     // RNG parameter

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

// NGT Parameters
struct NGTBuildParams {
    int edgeSizeForCreation = 10;
    int edgeSizeForSearch = 40;
    int edgeSizeLimitForCreation = 100;
    double insertionRadiusCoefficient = 1.1;
    int batchSizeForCreation = 200;
    int threadPoolSize = 1;

    // Others can be added if needed
};

// Build NGT index
inline std::unique_ptr<NGT::Index> build_ngt_index(
    const float* data, size_t n, size_t dim, const NGTBuildParams& params, const std::string& index_path) {
    NGT::Property property;
    property.dimension = dim;
    property.edgeSizeForCreation = params.edgeSizeForCreation;
    property.edgeSizeForSearch = params.edgeSizeForSearch;  // though usually 0 for creation?
    property.edgeSizeLimitForCreation = params.edgeSizeLimitForCreation;
    property.insertionRadiusCoefficient = params.insertionRadiusCoefficient;
    property.batchSizeForCreation = params.batchSizeForCreation;
    property.threadPoolSize = params.threadPoolSize;
    property.objectType = NGT::ObjectSpace::ObjectType::Float;
    property.distanceType = NGT::Index::Property::DistanceType::DistanceTypeL2;

    std::filesystem::create_directories(index_path);  // NGT creates a directory

    log("Creating NGT index at %s...\n", index_path.c_str());
    NGT::Index::create(index_path, property);

    auto index = std::make_unique<NGT::Index>(index_path);

    log("Appending data to NGT index...\n");
    for (size_t i = 0; i < n; i++) {
        std::vector<float> obj(data + i * dim, data + (i + 1) * dim);
        index->append(obj);
    }

    log("Building NGT index (createIndex)...\n");
    StopW stopw;
    index->createIndex(params.threadPoolSize);
    log("NGT Build time: %.2f seconds. Mem: %zu Mb. Peak: %zu Mb.\n",
        1e-6 * stopw.getElapsedTimeMicro(),
        getProcessCurrentRSS() / 1000000,
        getProcessPeakRSS() / 1000000);

    index->save();
    return index;
}

// Load existing NGT index or build new one
inline std::unique_ptr<NGT::Index> load_or_build_ngt_index(
    const Dataset& ds, const float* data, size_t n, size_t dim, const NGTBuildParams& params, const std::string& index_path) {
    std::unique_ptr<NGT::Index> index;

    // Check if index exists (it's a directory for NGT usually, or contains ngt files)
    // NGT::Index::create creates a directory.
    if (std::filesystem::exists(index_path) && std::filesystem::is_directory(index_path)) {
        log("Loading existing NGT index from %s\n", index_path.c_str());
        index = std::make_unique<NGT::Index>(index_path);
        log("Mem: %zu Mb. Peak: %zu Mb.\n", getProcessCurrentRSS() / 1000000, getProcessPeakRSS() / 1000000);
    } else {
        // Output directory is guaranteed by dataset logic usually, but let's be safe
        // But NGT::Index::create expects the path to be the directory to create.
        // So we ensure the PARENT exists.
        std::filesystem::path p(index_path);
        ensure_directory(p.parent_path());

        index = build_ngt_index(data, n, dim, params, index_path);
    }

    return index;
}

}  // namespace ngt::benchmark
