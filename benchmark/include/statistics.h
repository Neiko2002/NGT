#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
#include <vector>

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#include "NGT/Index.h"
#include "dataset.h"
#include "file_io.h"
#include "logging.h"
#include "util.h"

namespace ngt::benchmark::statistics {

// ============================================================================
// Helpers
// ============================================================================

static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out) {
    return ivecs_read(fname, d_out, n_out);
}

// Convert NGT graph to vector of vectors
static std::vector<std::vector<uint32_t>> getCompactGraph(NGT::GraphIndex& graphIndex) {
    const size_t size = graphIndex.repository.size();
    auto compact_graph = std::vector<std::vector<uint32_t>>(size);

    for (size_t id = 1; id < size; id++) {
        // NGT nodes start at 1 usually, but repository size includes 0 (dummy).
        try {
            const auto vertex = graphIndex.getNode(id);
            if (vertex) {
                auto& neighbor_ids = compact_graph[id];
                for (size_t i = 0; i < vertex->size(); i++) {
                    // (*vertex)[i] is NGT::ObjectDistance
                    auto neighbor_id = (*vertex)[i].id;
                    neighbor_ids.push_back(neighbor_id);
                }
            }
        } catch (...) {
            // ignore removed nodes or errors
        }
    }
    return compact_graph;
}

// ============================================================================
// Search Reachability (from seed nodes)
// ============================================================================

// Computes how many vertices are reachable from the default seed set.
static uint32_t compute_reachability_count(std::vector<std::vector<uint32_t>>& graph) {
    const auto graph_size = graph.size();
    if (graph_size == 0) return 0;

    unsigned L = 100;  // L_search equivalent
    unsigned seed = 1998;
    std::mt19937 rng(seed);
    std::vector<unsigned> init_ids(L);
    GenRandom(rng, init_ids.data(), L, static_cast<unsigned>(graph_size));

    std::vector<bool> visited(graph_size);
    std::vector<uint32_t> frontier;
    for (unsigned s : init_ids) {
        if (s < graph_size) {
            visited[s] = true;
            frontier.push_back(s);
        }
    }

    while (!frontier.empty()) {
        std::vector<uint32_t> next_frontier;
        for (uint32_t v : frontier) {
            for (uint32_t neighbor : graph[v]) {
                if (neighbor < graph_size && !visited[neighbor]) {  // Check bound just in case
                    visited[neighbor] = true;
                    next_frontier.push_back(neighbor);
                }
            }
        }
        frontier = std::move(next_frontier);
    }

    uint32_t count = 0;
    for (size_t i = 1; i < graph_size; i++) {  // id 0 is dummy in NGT
        if (visited[i]) count++;
    }
    log("Seed Reachability is %u out of %zu\n", count, graph_size - 1);
    return count;
}

// ============================================================================
// Exploration Reachability (average reachability from each vertex)
// ============================================================================

namespace {  // anonymous namespace - internal implementation

inline uint32_t popcount64(uint64_t x) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt64(x));
#else
    return static_cast<uint32_t>(__builtin_popcountll(x));
#endif
}

// Word-based bitset: bit i lives in words[i >> 6] at position (i & 63).
struct Bitset {
    size_t bit_count;
    std::vector<uint64_t> words;

    explicit Bitset(size_t n) : bit_count(n), words((n + 63) / 64, 0ull) {}

    bool test(uint32_t i) const { return (words[i >> 6] >> (i & 63)) & 1ull; }

    bool try_set(uint32_t i) {
        uint64_t& w = words[i >> 6];
        uint64_t mask = 1ull << (i & 63);
        bool was_unset = (w & mask) == 0;
        w |= mask;
        return was_unset;
    }

    uint32_t or_merge_count_new(const Bitset& other) {
        uint32_t added = 0;
        for (size_t i = 0; i < words.size(); i++) {
            uint64_t before = words[i];
            uint64_t new_bits = (~before) & other.words[i];
            words[i] = before | other.words[i];
            added += popcount64(new_bits);
        }
        return added;
    }
};

// Cached reachability entry (immutable after publication).
struct CacheEntry {
    uint32_t reach_count;
    Bitset reachable;
    CacheEntry(uint32_t rc, Bitset&& bs) : reach_count(rc), reachable(std::move(bs)) {}
};

// Thread-safe reachability cache.
struct ReachCache {
    static constexpr uint32_t NO_ENTRY = UINT32_MAX;

    std::vector<std::atomic<uint32_t>> mapping;
    std::vector<CacheEntry> entries;
    std::atomic<size_t> published{0};
    std::mutex write_mutex;

    explicit ReachCache(size_t n) : mapping(n) {
        for (auto& a : mapping) a.store(NO_ENTRY, std::memory_order_relaxed);
        entries.reserve(n);
    }

    // Returns {index, entry*} or {NO_ENTRY, nullptr}.
    std::pair<uint32_t, const CacheEntry*> lookup(uint32_t v) const {
        uint32_t idx = mapping[v].load(std::memory_order_acquire);
        if (idx == NO_ENTRY) return {NO_ENTRY, nullptr};
        if (idx >= published.load(std::memory_order_acquire)) return {NO_ENTRY, nullptr};
        return {idx, &entries[idx]};
    }

    void alias(uint32_t v, uint32_t idx) { mapping[v].store(idx, std::memory_order_release); }

    uint32_t store(uint32_t v, uint32_t reach_count, Bitset&& bs) {
        std::lock_guard<std::mutex> lock(write_mutex);
        size_t pos = published.load(std::memory_order_relaxed);
        entries.emplace_back(reach_count, std::move(bs));
        published.store(pos + 1, std::memory_order_release);
        mapping[v].store(static_cast<uint32_t>(pos), std::memory_order_release);
        return static_cast<uint32_t>(pos);
    }
};

// BFS with memoization for a single start vertex.
template <typename BestTracker>
uint32_t compute_reach_for_vertex(size_t start, std::vector<std::vector<uint32_t>>& graph, ReachCache& cache, BestTracker& best_global) {
    const size_t n = graph.size();

    Bitset visited(n);
    visited.try_set(static_cast<uint32_t>(start));
    uint32_t reach = 1;

    std::vector<uint32_t> frontier;
    frontier.reserve(64);
    frontier.push_back(static_cast<uint32_t>(start));

    uint32_t best_cache_idx = ReachCache::NO_ENTRY;
    uint32_t best_cache_reach = 0;

    while (!frontier.empty() && best_cache_reach < n) {
        std::vector<uint32_t> next;
        for (size_t i = 0; i < frontier.size() && best_cache_reach < n; i++) {
            for (uint32_t neighbor : graph[frontier[i]]) {
                if (neighbor < n && !visited.test(neighbor)) {  // check bounds
                    if (visited.try_set(neighbor)) {
                        next.push_back(neighbor);
                        reach++;
                    }

                    auto [idx, entry] = cache.lookup(neighbor);
                    if (entry && entry->reach_count > best_cache_reach) {
                        best_cache_idx = idx;
                        best_cache_reach = entry->reach_count;

                        if (entry->reach_count == n) break;
                        reach += visited.or_merge_count_new(entry->reachable);
                    }
                }
            }
        }
        frontier = std::move(next);
    }

    // Full graph reached via cache hit?
    if (best_cache_reach == n) {
        cache.alias(static_cast<uint32_t>(start), best_cache_idx);
        return static_cast<uint32_t>(n);
    }

    // Update cache: promote if new best, alias if we used a cache, else seed.
    if (best_global.try_promote(reach)) {
        cache.store(static_cast<uint32_t>(start), reach, std::move(visited));
    } else if (best_cache_reach > 0) {
        cache.alias(static_cast<uint32_t>(start), best_cache_idx);
    } else {
        cache.store(static_cast<uint32_t>(start), reach, std::move(visited));
    }

    return reach;
}

struct BestSingleThread {
    uint32_t value = 0;
    bool try_promote(uint32_t r) {
        if (r > value) {
            value = r;
            return true;
        }
        return false;
    }
};

struct BestMultiThread {
    std::atomic<uint32_t> value{0};
    bool try_promote(uint32_t r) {
        uint32_t cur = value.load(std::memory_order_relaxed);
        while (r > cur) {
            if (value.compare_exchange_weak(cur, r)) return true;
        }
        return false;
    }
};

}  // anonymous namespace

// Computes mean exploration reachability over all vertices.
// num_threads=1: single-threaded, num_threads>1: parallel with shared cache.
static uint32_t compute_avg_reach(std::vector<std::vector<uint32_t>>& graph, size_t num_threads = 1) {
    const size_t n = graph.size();
    if (n == 0) return 0;

    // NGT 0 is dummy
    size_t actual_n = n - 1;

    ReachCache cache(n);

    if (num_threads <= 1) {
        BestSingleThread best;
        uint64_t total = 0;
        for (size_t v = 1; v < n; v++) {
            total += compute_reach_for_vertex(v, graph, cache, best);
        }
        return static_cast<uint32_t>(total / actual_n);
    }

    // Parallel
    num_threads = std::min(num_threads, n);
    BestMultiThread best;
    std::vector<uint64_t> sums(num_threads, 0);

    parallel_for(size_t{1}, n, num_threads, [&](size_t v, size_t tid) { sums[tid] += compute_reach_for_vertex(v, graph, cache, best); });

    uint64_t total = 0;
    for (auto s : sums) total += s;
    return static_cast<uint32_t>(total / actual_n);
}

// ============================================================================
// Full Graph Statistics
// ============================================================================

static void compute_stats(NGT::Index* index, const char* top_list_file) {
    log("\n--- Graph Analysis ---\n");

    NGT::GraphIndex& graphIndex = (NGT::GraphIndex&)index->getIndex();
    auto graph = getCompactGraph(graphIndex);  // This copies the graph to std::vector<std::vector<uint32_t>>

    // NGT stats: id 0 is dummy. Size is N+1.
    const size_t graph_size = graph.size();
    const size_t num_vertices = graph_size - 1;

    // --- Graph Quality ---
    log("Computing graph quality...\n");
    float perfect_neighbor_ratio = 0.0f;
    bool gq_available = false;
    {
        size_t top_list_dims = 0, top_list_count = 0;
        auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
        if (!all_top_list) {
            log("Skipping graph quality: could not load TopList file %s\n", top_list_file);
        } else if (top_list_count != num_vertices) {
            log("Skipping graph quality: TopList element count mismatch: %zu vs %zu\n", top_list_count, num_vertices);
        } else if (top_list_dims < 1) {
            log("Skipping graph quality: TopList has invalid k=%zu\n", top_list_dims);
        } else {
            uint64_t perfect = 0, total = 0;
            for (size_t n = 1; n < graph_size; n++) {  // 1-indexed
                auto& neighbors = graph[n];
                // Check against toplist which is 0-indexed usually (but need to check if file has 0 or 1 based ids)
                // usually .ivecs are 0 based. NGT graph has 1 based ids?
                // In getCompactGraph from l2-float_stats.cpp, it doesn't adjust.
                // But in compute_stats in l2-float_stats.cpp:
                // neighbor_index = neighbor_indizies[e] - 1; // ids start at 1 not 0
                // So yes, we need to subtract 1.

                size_t k = std::min(neighbors.size(), top_list_dims);
                // top_list for node n (where n is 1-based, so index n-1)
                const auto* top = all_top_list.get() + (n - 1) * top_list_dims;

                total += k;
                for (size_t e = 0; e < k; e++) {
                    uint32_t neighbor_0 = neighbors[e] - 1;
                    for (size_t i = 0; i < k; i++) {
                        if (neighbor_0 == top[i]) {
                            perfect++;
                            break;
                        }
                    }
                }
            }
            if (total > 0) {
                perfect_neighbor_ratio = static_cast<float>(perfect) / static_cast<float>(total);
                gq_available = true;
            }
        }
    }

    // --- Degree Statistics ---
    size_t min_out = SIZE_MAX, max_out = 0, total_edges = 0;
    for (size_t n = 1; n < graph_size; n++) {
        size_t deg = graph[n].size();
        min_out = std::min(min_out, deg);
        max_out = std::max(max_out, deg);
        total_edges += deg;
    }
    if (min_out == SIZE_MAX) min_out = 0;

    std::vector<uint32_t> in_degree(graph_size, 0);
    for (size_t n = 1; n < graph_size; n++) {
        for (uint32_t neighbor : graph[n]) {
            if (neighbor < graph_size) in_degree[neighbor]++;
        }
    }
    uint32_t min_in = UINT32_MAX, max_in = 0, source_nodes = 0;
    for (size_t n = 1; n < graph_size; n++) {
        min_in = std::min(min_in, in_degree[n]);
        max_in = std::max(max_in, in_degree[n]);
        if (in_degree[n] == 0) source_nodes++;
    }
    if (min_in == UINT32_MAX) min_in = 0;

    // --- Reachability ---
    log("Computing search reachability...\n");
    uint32_t search_reach = compute_reachability_count(graph);

    log("Computing exploration reachability...\n");
    uint32_t explore_reach = compute_avg_reach(graph);

    // --- Output ---
    double avg_deg = (num_vertices > 0) ? static_cast<double>(total_edges) / num_vertices : 0.0;
    double search_pct = (num_vertices > 0) ? 100.0 * search_reach / num_vertices : 0.0;
    double explore_pct = (num_vertices > 0) ? 100.0 * explore_reach / num_vertices : 0.0;

    log("Graph Statistics:\n");
    log("  Vertices: %zu\n", num_vertices);
    log("  Total edges: %zu\n", total_edges);
    log("  Avg edges per vertex: %.0f\n", avg_deg);
    log("  Out-degree: avg=%.2f, min=%zu, max=%zu\n", avg_deg, min_out, max_out);
    log("  In-degree:  avg=%.2f, min=%u, max=%u, source_nodes=%u\n", avg_deg, min_in, max_in, source_nodes);
    log("  Graph Quality (GQ): %s\n", gq_available ? std::to_string(perfect_neighbor_ratio).c_str() : "N/A");
    log("  Search Reachability: %.2f%%\n", search_pct);
    log("  Exploration Reachability: %.2f%%\n", explore_pct);
}

}  // namespace ngt::benchmark::statistics
