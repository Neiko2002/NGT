#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "index_ssg.h"
#include "logging.h"
#include "parameters.h"
#include "stopwatch.h"

namespace ssg::benchmark {

// ANNS test with varying L_search parameters
// ANNS test with varying L_search parameters
template <typename T>
static void test_graph_anns(T* graph,
                            const float* base_data,
                            const float* query_data,
                            size_t query_count,
                            size_t dim,
                            const std::vector<std::vector<uint32_t>>& ground_truth,
                            const uint32_t repeat,
                            const uint32_t k,
                            const std::vector<unsigned>& L_search_params,
                            const float recall_target = 0.995f) {
    std::vector<unsigned> L_sorted = L_search_params;
    std::sort(L_sorted.begin(), L_sorted.end());

    // Create result buffer
    std::vector<unsigned> result(k);

    for (unsigned L_search : L_sorted) {
        // L_search must be >= k
        if (L_search < k) continue;

        efanna2e::Parameters params;
        params.Set<unsigned>("L_search", L_search);

        StopW stopw;
        size_t correct = 0;

        for (uint32_t t = 0; t < repeat; t++) {
            for (size_t i = 0; i < query_count; ++i) {
                std::fill(result.begin(), result.end(), 0u);

                graph->Search(query_data + i * dim, base_data, k, params, result.data());

                // Compare with ground truth
                if (i < ground_truth.size()) {
                    const auto& gt = ground_truth[i];
                    for (size_t r = 0; r < k; r++) {
                        if (std::binary_search(gt.begin(), gt.end(), result[r])) {
                            correct++;
                        }
                    }
                }
            }
        }

        float recall = static_cast<float>(correct) / static_cast<float>(repeat) / (static_cast<float>(query_count) * static_cast<float>(k));
        auto time_us_per_query = stopw.getElapsedTimeMicro() / (query_count * repeat);

        log("L_search %5u, recall %.4f, time_us_per_query %8lld\n", L_search, recall, static_cast<long long>(time_us_per_query));

        // Early exit if recall target reached
        if (recall >= recall_target) {
            log("Recall target %.3f reached, stopping L_search sweep\n", recall_target);
            break;
        }
    }
}

// Exploration test (from entry nodes)
template <typename T>
static void test_graph_explore(T* graph,
                               const float* base_data,
                               const float* query_data,
                               size_t query_count,
                               size_t dim,
                               const std::vector<std::vector<uint32_t>>& ground_truth,
                               const std::vector<std::vector<uint32_t>>& entry_node_indices,
                               const uint32_t k,
                               const float recall_target = 0.995f) {
    log("Testing Exploration (k=%u)...\n", k);

    if (entry_node_indices.size() != query_count) {
        log("Exploration Test aborted: entry_node_indices size (%zu) != queries size (%zu)\n", entry_node_indices.size(), query_count);
        return;
    }

    for (size_t q = 0; q < entry_node_indices.size(); ++q) {
        if (entry_node_indices[q].empty()) {
            log("Exploration Test aborted: entry_node_indices[%zu] is empty\n", q);
            return;
        }
    }

    std::vector<unsigned> result(k);
    float last_recall = -1.0f;

    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= 2; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i - 1)) : (k_factor * i));

            StopW stopw;
            size_t correct = 0;

            for (size_t q = 0; q < query_count; ++q) {
                std::fill(result.begin(), result.end(), 0u);

                unsigned initial_node = entry_node_indices[q][0];
                graph->Explore(initial_node, query_data + q * dim, base_data, k, result.data(), max_distance_count);

                if (q < ground_truth.size()) {
                    const auto& gt = ground_truth[q];
                    for (size_t r = 0; r < k; r++) {
                        if (std::binary_search(gt.begin(), gt.end(), result[r])) {
                            correct++;
                        }
                    }
                }
            }

            const float denom = static_cast<float>(query_count) * static_cast<float>(k);
            const float recall = denom > 0.0f ? static_cast<float>(correct) / denom : 0.0f;
            const uint64_t time_us_per_query = query_count > 0 ? (stopw.getElapsedTimeMicro() / query_count) : 0;

            log("k %5u, max_distance_count %6u, recall %.4f, time_us_per_query %6llu\n",
                k,
                max_distance_count,
                recall,
                static_cast<unsigned long long>(time_us_per_query));

            if (recall == last_recall) {
                log("Recall stabilized at %.4f, stopping exploration sweep\n", recall);
                return;
            }
            last_recall = recall;

            // Early exit if recall target reached
            if (recall >= recall_target) {
                log("Recall target %.3f reached, stopping exploration sweep\n", recall_target);
                return;
            }
        }
    }
}

}  // namespace ssg::benchmark
