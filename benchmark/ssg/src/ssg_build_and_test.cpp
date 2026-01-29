#ifndef SSG_VALUE_TYPE
    #define SSG_VALUE_TYPE float
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "benchmark.h"
#include "build.h"
#include "dataset.h"
#include "logging.h"
#include "statistics.h"

using namespace ssg::benchmark;

struct DatasetConfig {
    DatasetName dataset_name = DatasetName::SIFT1M;
    Metric metric = Metric::L2;

    CreateGraphParams create_graph;  // Used for KNN graph filename generation
    SSGBuildParams ssg;
};

static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;

    if (dataset_name == DatasetName::SIFT1M) {
        // KNN params (for input filename)
        conf.create_graph.K = 200;
        conf.create_graph.L = 200;
        conf.create_graph.iter = 12;
        conf.create_graph.S = 10;
        conf.create_graph.R = 100;
        conf.create_graph.L_search = {100, 120, 140, 170, 200, 300, 500};

        // SSG params
        conf.ssg.L = 100;
        conf.ssg.R = 50;
        conf.ssg.A = 60;
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.create_graph.K = 200;
        conf.create_graph.L = 200;
        conf.create_graph.iter = 12;
        conf.create_graph.S = 10;
        conf.create_graph.R = 100;
        conf.create_graph.nTrees = 0;
        conf.create_graph.mLevel = 0;
        conf.create_graph.L_search = {100, 150, 200, 300, 600};

        conf.ssg.L = 100;
        conf.ssg.R = 50;
        conf.ssg.A = 60;
    } else if (dataset_name == DatasetName::GLOVE) {
        conf.create_graph.K = 400;
        conf.create_graph.L = 420;
        conf.create_graph.iter = 12;
        conf.create_graph.S = 15;
        conf.create_graph.R = 200;
        conf.create_graph.L_search = {500, 1000, 1500, 2000, 3000, 4000};

        conf.ssg.L = 500;
        conf.ssg.R = 50;
        conf.ssg.A = 60;
    } else if (dataset_name == DatasetName::ENRON) {
        conf.create_graph.K = 100;
        conf.create_graph.L = 110;
        conf.create_graph.iter = 7;
        conf.create_graph.S = 20;
        conf.create_graph.R = 300;
        conf.create_graph.L_search = {100, 120, 140, 170, 200, 300};
        conf.create_graph.anns_repeat = 5;

        conf.ssg.L = 300;
        conf.ssg.R = 30;
        conf.ssg.A = 60;
    } else if (dataset_name == DatasetName::AUDIO) {
        conf.create_graph.K = 400;
        conf.create_graph.L = 400;
        conf.create_graph.iter = 5;
        conf.create_graph.S = 25;
        conf.create_graph.R = 200;
        conf.create_graph.nTrees = 0;
        conf.create_graph.mLevel = 0;
        conf.create_graph.L_search = {100, 200, 300, 500};
        conf.create_graph.anns_repeat = 10;

        conf.ssg.L = 50;
        conf.ssg.R = 20;
        conf.ssg.A = 60;
    }

    return conf;
}

struct GraphPaths {
    std::filesystem::path ssg_dir;
    std::filesystem::path efanna_dir;

    GraphPaths(const Dataset& ds) : ssg_dir(ds.data_root() / ds.name() / "ssg"), efanna_dir(ds.data_root() / ds.name() / "efanna") {}

    // KNN graph base name (copied from efanna logic)
    std::string knn_base_name(const CreateGraphParams& cg) const {
        return string_format("K%u_L%u_It%u_S%u_R%u_(nTrees%u_mLevel%u)", cg.K, cg.L, cg.iter, cg.S, cg.R, cg.nTrees, cg.mLevel);
    }

    std::string ssg_base_name(const CreateGraphParams& cg, const SSGBuildParams& ssg) const {
        return string_format("L%u_R%u_A%u_efa%s", ssg.L, ssg.R, ssg.A, knn_base_name(cg).c_str());
    }

    std::string graph_directory() const { return ssg_dir.string(); }

    std::string efanna_file(const CreateGraphParams& cg) const { return (efanna_dir / (knn_base_name(cg) + ".efa")).string(); }

    std::string ssg_file(const CreateGraphParams& cg, const SSGBuildParams& ssg) const {
        return (ssg_dir / (ssg_base_name(cg, ssg) + ".ssg")).string();
    }

    std::string graph_log_file(const CreateGraphParams& cg, const SSGBuildParams& ssg) const {
        return (ssg_dir / (ssg_base_name(cg, ssg) + ".log")).string();
    }
};

static void run_graph_stats(efanna2e::IndexSSG* index, const Dataset& ds, bool use_half_gt) {
    const std::string gt_file = ds.base_groundtruth_file(use_half_gt);

    if (std::filesystem::exists(gt_file)) {
        statistics::compute_stats(index, gt_file.c_str());
    } else {
        log("Skipping stats: ground truth file not found %s\n", gt_file.c_str());
    }
}

static void run_anns_test(efanna2e::Index* index,
                          const float* base_data,
                          const float* query_data,
                          size_t query_count,
                          size_t dim,
                          const Dataset& ds,
                          const CreateGraphParams& cg,
                          bool use_half_gt) {
    auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
    wait_before_test();

    test_graph_anns(index, base_data, query_data, query_count, dim, ground_truth, cg.anns_repeat, cg.anns_k, cg.L_search);
}

static void run_explore_test(efanna2e::IndexSSG* index,
                             const Dataset& ds,
                             const float* base_data,
                             const float* query_data,
                             size_t query_count,
                             size_t dim,
                             const CreateGraphParams& cg,
                             bool use_half_gt) {
    std::string entry_file = ds.explore_entry_vertex_file();
    const std::string explore_gt_file = ds.explore_groundtruth_file(use_half_gt);
    std::string explore_query_file = ds.explore_query_file();

    // Check if all required files exist
    if (!std::filesystem::exists(entry_file)) {
        log("Skipping exploration test: entry file not found: %s\n", entry_file.c_str());
        return;
    }
    if (!std::filesystem::exists(explore_gt_file)) {
        log("Skipping exploration test: ground truth file not found: %s\n", explore_gt_file.c_str());
        return;
    }
    if (!std::filesystem::exists(explore_query_file)) {
        log("Skipping exploration test: explore query file not found: %s\n", explore_query_file.c_str());
        return;
    }

    // Load exploration queries
    log("Loading exploration queries from: %s\n", explore_query_file.c_str());
    LoadedData explore_queries = ds.load_explore_query();
    log("Loaded %u exploration queries with %u dimensions\n", explore_queries.num, explore_queries.dim);

    // Load entry indices
    size_t entry_count = 0;
    auto entry_indices = load_ivecs_as_vectors(entry_file.c_str(), entry_count);

    // Load ground truth
    size_t dim_gt = 0, n_gt = 0;
    auto gt_ptr = ivecs_read(explore_gt_file.c_str(), dim_gt, n_gt);
    if (!gt_ptr) {
        log("Failed to load ground truth\n");
        return;
    }

    std::vector<std::vector<uint32_t>> explore_gt_vec(n_gt);
    for (size_t i = 0; i < n_gt; ++i) {
        explore_gt_vec[i].assign(gt_ptr.get() + i * dim_gt, gt_ptr.get() + (i + 1) * dim_gt);
        std::sort(explore_gt_vec[i].begin(), explore_gt_vec[i].end());
    }

    // Verify dimensions match
    if (explore_queries.num != entry_indices.size() || explore_queries.num != explore_gt_vec.size()) {
        log("Warning: dimension mismatch - queries=%u, entries=%zu, gt=%zu\n",
            explore_queries.num,
            entry_indices.size(),
            explore_gt_vec.size());
    }

    wait_before_test();
    test_graph_explore(
        index, base_data, explore_queries.data, explore_queries.num, explore_queries.dim, explore_gt_vec, entry_indices, cg.explore_k);
}

static void run_create_graph_test(const Dataset& ds,
                                  const CreateGraphParams& cg,
                                  const SSGBuildParams& ssg,
                                  const GraphPaths& paths,
                                  const LoadedData& base_data,
                                  const LoadedData& query_data,
                                  const std::string& test_name) {
    std::string graph_path = paths.ssg_file(cg, ssg);
    std::string log_path = paths.graph_log_file(cg, ssg);
    std::string knn_path = paths.efanna_file(cg);

    // Check if KNN graph exists
    if (!std::filesystem::exists(knn_path)) {
        log("Error: KNN graph not found at %s. Please build it first or check path.\n", knn_path.c_str());
        return;
    }

    std::filesystem::create_directories(paths.graph_directory());
    if (std::filesystem::exists(log_path)) {
        log("CREATE_GRAPH: Skipping - log file already exists: %s\n", log_path.c_str());
        return;
    }
    set_log_file(log_path, true);
    attach_cerr_to_log();
    attach_cout_to_log();

    log("\n=== %s Test ===\n", test_name.c_str());
    log("KNN Input: %s\n", knn_path.c_str());
    log("Settings: L=%u, R=%u, A=%u\n", ssg.L, ssg.R, ssg.A);
    log("Graph: %s\n", graph_path.c_str());
    log("Log: %s\n", log_path.c_str());
#ifdef _OPENMP
    log("Threads: %d\n", omp_get_max_threads());
#else
    log("Threads: 1 (OpenMP disabled)\n");
#endif

    log("Base data: size=%u, dim=%u\n", base_data.num, base_data.dim);
    log("Query data: size=%u, dim=%u\n", query_data.num, query_data.dim);
    log("Memory usage before build: %zu Mb, Peak memory usage: %zu Mb\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    auto index = load_or_build_ssg_index(ds, base_data.data, base_data.num, base_data.dim, ssg, knn_path, graph_path);

    if (index) {
        run_graph_stats(index.get(), ds, false);

        log("\n--- ANNS Test (k=%u) ---\n", cg.anns_k);
        run_anns_test(index.get(), base_data.data, query_data.data, query_data.num, query_data.dim, ds, cg, false);
        log("ANNS Test complete\n");

        log("\n--- Exploration Test (k=%u) ---\n", cg.explore_k);
        run_explore_test(index.get(), ds, base_data.data, query_data.data, query_data.num, query_data.dim, cg, false);
        log("Exploration Test complete\n");
    }

    reset_log_to_console();
    log("%s: Log written to: %s\n", test_name.c_str(), log_path.c_str());
}

int main(int argc, char** argv) {
    log("Testing SSG Graph...\n");

#if defined(__AVX__)
    std::cout << "use AVX2  ..." << std::endl;
#elif defined(__SSE2__)
    std::cout << "use SSE  ..." << std::endl;
#else
    std::cout << "use arch  ..." << std::endl;
#endif
    std::cout << "DATA_ALIGN_FACTOR " << DATA_ALIGN_FACTOR << std::endl;

#ifdef _OPENMP
    omp_set_dynamic(0);      // Explicitly disable dynamic teams
    omp_set_num_threads(1);  // Use 1 threads for all consecutive parallel regions

    std::cout << "_OPENMP " << omp_get_max_threads() << " threads (max)" << std::endl;
#endif

    const auto data_path = std::filesystem::path(DATA_PATH);
    log("data_path %s\n", data_path.string().c_str());

    DatasetName ds_name = DatasetName::AUDIO;
    std::string data_root = data_path.string();
    bool do_run = true;

    if (data_root.empty()) {
        log("WARNING: DATA_PATH is empty! Please provide it as a command line "
            "argument or set it in CMake.\n");
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "help" || arg == "--help") {
            log("Usage: ssg_build_and_test <dataset> [data_root] [--run|--dry-run]\n");
            log("Datasets: sift1m, deep1m, audio, glove, enron, all\n");
            log("Options: [data_root] path (default: DATA_PATH), --run or "
                "--dry-run\n");
            return 0;
        }

        if (arg == "--run") {
            do_run = true;
            continue;
        }
        if (arg == "--dry-run") {
            do_run = false;
            continue;
        }

        auto parsed_ds = DatasetName::from_string(arg);
        if (parsed_ds.is_valid()) {
            ds_name = parsed_ds;
            continue;
        }

        if (arg == "create_graph") {
            // keep it for compatibility or remove if not needed
            continue;
        }

        data_root = arg;
    }

    auto run_for_dataset = [&](const DatasetName& name) -> int {
        try {
            Dataset ds(name, data_root);
            auto config = get_dataset_config(name);
            config.metric = ds.info().metric;
            GraphPaths graph_paths(ds);

            log("\n=== Dataset: %s ===\n", ds.name());
            log("Repository file: %s\n", ds.base_file().c_str());
            log("Query file: %s\n", ds.query_file().c_str());
            log("Graph directory: %s\n", graph_paths.graph_directory().c_str());
            log("Query GT (full): %s\n", ds.query_groundtruth_file_full().c_str());
            log("Query GT (half): %s\n", ds.query_groundtruth_file_half().c_str());
            log("Base GT (full): %s\n", ds.base_groundtruth_file(false).c_str());
            log("Explore GT (full): %s\n", ds.explore_groundtruth_file(false).c_str());

            if (do_run) {
                if (!std::filesystem::exists(ds.base_file())) {
                    log("Missing base file: %s\n", ds.base_file().c_str());
                    return 1;
                }
                if (!std::filesystem::exists(ds.query_file())) {
                    log("Missing query file: %s\n", ds.query_file().c_str());
                    return 1;
                }
                if (!std::filesystem::exists(ds.query_groundtruth_file_full())) {
                    log("Missing query groundtruth (full): %s\n", ds.query_groundtruth_file_full().c_str());
                    return 1;
                }
                if (!std::filesystem::exists(ds.query_groundtruth_file_half())) {
                    log("Missing query groundtruth (half): %s\n", ds.query_groundtruth_file_half().c_str());
                    return 1;
                }

                log("\nLoading data...\n");
                auto base_data = ds.load_base();
                auto query_data = ds.load_query();

                run_create_graph_test(ds, config.create_graph, config.ssg, graph_paths, base_data, query_data, "CREATE_GRAPH");
            }

            return 0;
        } catch (const std::exception& e) {
            log("ERROR: Dataset '%s' failed with exception: %s\n", name.name(), e.what());
            return 1;
        } catch (...) {
            log("ERROR: Dataset '%s' failed with unknown exception\n", name.name());
            return 1;
        }
    };

    if (ds_name == DatasetName::ALL) {
        for (const auto& name : DatasetName::all()) {
            log("\n--- ALL: starting dataset %s ---\n", name.name());
            int rc = run_for_dataset(name);
            log("--- ALL: finished dataset %s with rc=%d ---\n", name.name(), rc);
        }
    } else {
        return run_for_dataset(ds_name);
    }

    return 0;
}
