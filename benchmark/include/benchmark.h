#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "NGT/Index.h"
#include "logging.h"
#include "stopwatch.h"

namespace ngt::benchmark {

// ANNS test with varying epsilon
// ANNS test with varying epsilons
static void test_ngt_anns(NGT::Index& index,
                          const float* query_data,
                          size_t query_count,
                          size_t dim,
                          const std::vector<std::vector<uint32_t>>& ground_truth,
                          const uint32_t k,
                          const std::vector<float>& epsilons,
                          int repeat = 1,
                          float recall_target = 0.995f,
                          int edge_size = -1 /* default */) {
    for (float epsilon : epsilons) {
        long long total_time_us = 0;
        float recall = 0.0f;

        for (int r = 0; r < repeat; ++r) {
            StopW stopw;
            size_t correct = 0;

            for (size_t i = 0; i < query_count; ++i) {
                std::vector<float> query(query_data + i * dim, query_data + (i + 1) * dim);
                NGT::SearchQuery sc(query);
                NGT::ObjectDistances result;
                sc.setResults(&result);
                sc.setSize(k);
                sc.setEpsilon(epsilon);
                sc.setEdgeSize(edge_size);

                try {
                    index.search(sc);
                } catch (NGT::Exception& err) {
                    continue;
                }

                // Count correct results
                if (i < ground_truth.size()) {
                    const auto& gt = ground_truth[i];
                    for (const auto& res : result) {
                        if (res.id > 0) {
                            if (std::binary_search(gt.begin(), gt.end(), res.id - 1)) {
                                correct++;
                            }
                        }
                    }
                }
            }
            total_time_us += stopw.getElapsedTimeMicro();
            if (r == 0) {
                recall = static_cast<float>(correct) / (static_cast<float>(query_count) * static_cast<float>(k));
            }
        }

        auto time_us_per_query = (total_time_us / repeat) / query_count;

        log("Epsilon %.2f, Edge %d, Recall %.4f, Time %lld us/query\n",
            epsilon,
            edge_size,
            recall,
            static_cast<long long>(time_us_per_query));

        if (recall >= recall_target) {
            log("Recall target %.3f reached, stopping epsilon sweep\n", recall_target);
            break;
        }
    }
}

// Exploration test
static void test_ngt_explore(NGT::Index& index,
                             const float* query_data,
                             size_t query_count,
                             size_t dim,
                             const std::vector<std::vector<uint32_t>>& ground_truth,
                             const std::vector<uint32_t>& entry_nodes,
                             int k,
                             float epsilon,
                             float recall_target = 0.995f) {
    if (query_count > entry_nodes.size()) {
        log("Error: Not enough entry nodes (%zu) for queries (%zu)\n", entry_nodes.size(), query_count);
        return;
    }

    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= 2; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i - 1)) : (k_factor * i));

            StopW stopw;
            size_t correct = 0;
            size_t empty_list = 0;
            size_t short_list = 0;

            for (size_t q = 0; q < query_count; q++) {
                auto entry_node_index = (uint32_t)(entry_nodes[q] + 1);  // 0-based to 1-based
                std::vector<float> query(query_data + q * dim, query_data + (q + 1) * dim);

                NGT::SearchQuery sc(query);
                NGT::ObjectDistances objects;
                sc.setResults(&objects);
                sc.setSize(k);
                sc.setEpsilon(epsilon);

                try {
                    index.explore(sc, entry_node_index, max_distance_count);
                } catch (NGT::Exception& err) {
                    continue;
                }

                // compare
                if (q < ground_truth.size()) {
                    const auto& gt = ground_truth[q];
                    for (size_t r = 0; r < objects.size(); r++) {
                        // objects[r].id is 1-based, gt is 0-based
                        if (std::binary_search(gt.begin(), gt.end(), objects[r].id - 1)) {
                            correct++;
                        }
                    }
                }

                if (objects.size() < k) short_list++;
                if (objects.size() == 0) empty_list++;
            }

            auto time_us_per_query = stopw.getElapsedTimeMicro() / query_count;
            double recall = (double)correct / (query_count * k);

            log("Explore: MaxDist %6d, Recall %.4f, Time %lld us/query, Short %zu\n",
                max_distance_count,
                recall,
                (long long)time_us_per_query,
                short_list);

            if (recall >= recall_target) {
                log("Recall target %.3f reached, stopping exploration sweep\n", recall_target);
                return;
            }
        }
    }
}

}  // namespace ngt::benchmark
