#ifndef NGT_VALUE_TYPE
    #define NGT_VALUE_TYPE float
#endif
#define NOMINMAX

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "NGT/GraphOptimizer.h"
#include "NGT/Index.h"
#include "benchmark.h"
#include "dataset.h"
#include "logging.h"
#include "statistics.h"
#include "util.h"

// Transitive includes: filesystem (via dataset.h)

using namespace ngt::benchmark;

struct ANNGParams {
    int edgeSizeForCreation = 200;
    int edgeSizeLimitForCreation = 200;
    double insertionRadiusCoefficient = 1.1;
    int outgoingEdge = 10;
    int incomingEdge = 120;
    int batchSizeForCreation = 500;
};

struct ONNGParams {
    int numOfOutgoingEdges = 10;
    int numOfIncomingEdges = 120;
    int minNumOfEdges = 0;
    double gtEpsilon = 0.1;
    double margin = 0.2;
    int numOfResults = 100;
    int numOfQueries = 100;
    bool varianceControl = false;  // Not explicitly in l2-float but part of optimizer
};

struct DatasetConfig {
    DatasetName dataset_name = DatasetName::SIFT1M;
    ANNGParams anng;
    ONNGParams onng;

    // ANNS Test Params
    int anns_k = 100;
    int anns_repeat = 1;
    std::vector<float> epsilons = {0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20};
    std::vector<int> edge_sizes = {-1};  // -1 means default

    // Exploration Params
    int explore_k = 1000;
    float explore_epsilon = 0.03f;
};

static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;

    if (dataset_name == DatasetName::SIFT1M) {
        // NGT repo SIFT1M
        conf.anng.edgeSizeLimitForCreation = 100;    // same as edgeSizeForCreation
        conf.anng.edgeSizeForCreation = 100;         // kc with kc > eo and kc > ei
        conf.anng.insertionRadiusCoefficient = 1.1;  // eps_c´= 0.1
        conf.anng.outgoingEdge = 10;                 // eo
        conf.anng.incomingEdge = 120;                // ei

        // NGT repo for SIFT1M
        // https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml#L378
        // https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/onng_ngt.py
        // https://github.com/yahoojapan/NGT/tree/main/bin/ngt
        conf.onng.numOfOutgoingEdges = 10;
        conf.onng.numOfIncomingEdges = 120;

        // ANNS Test
        conf.anns_k = 100;
        conf.epsilons = {0.01f, 0.02f, 0.05f, 0.1f, 0.15f, 0.2f};
        conf.explore_k = 1000;

    } else if (dataset_name == DatasetName::DEEP1M) {
        // Deep1M, same as ONNG Paper Table 3 for SIFT1M
        conf.anng.edgeSizeLimitForCreation = 200;
        conf.anng.edgeSizeForCreation = 200;
        conf.anng.insertionRadiusCoefficient = 1.1;
        conf.anng.outgoingEdge = 30;
        conf.anng.incomingEdge = 110;

        // Deep1M, same as ONNG Paper Table 3 for SIFT1M
        conf.onng.numOfOutgoingEdges = 30;
        conf.onng.numOfIncomingEdges = 110;

        conf.anns_k = 100;
        conf.explore_k = 1000;

    } else if (dataset_name == DatasetName::GLOVE) {
        // ONNG Paper Table 3 on GloVe
        conf.anng.edgeSizeLimitForCreation = 200;
        conf.anng.edgeSizeForCreation = 200;
        conf.anng.insertionRadiusCoefficient = 1.1;
        conf.anng.outgoingEdge = 15;
        conf.anng.incomingEdge = 155;

        // ONNG Paper for GloVe
        conf.onng.numOfOutgoingEdges = 15;
        conf.onng.numOfIncomingEdges = 155;

        conf.anns_k = 100;
        conf.explore_k = 1000;

    } else if (dataset_name == DatasetName::ENRON) {
        // Enron (Weaves ANNG params)
        conf.anng.edgeSizeLimitForCreation = 200;
        conf.anng.edgeSizeForCreation = 200;
        conf.anng.insertionRadiusCoefficient = 1.1;
        conf.anng.outgoingEdge = 20;
        conf.anng.incomingEdge = 100;

        // WEAVES on Enron
        conf.onng.numOfOutgoingEdges = 20;
        conf.onng.numOfIncomingEdges = 100;

        conf.anns_k = 100;
        conf.anns_repeat = 10;
        conf.explore_k = 1000;
        conf.explore_epsilon = 0.012f;

    } else if (dataset_name == DatasetName::AUDIO) {
        // Audio
        conf.anng.edgeSizeLimitForCreation = 200;
        conf.anng.edgeSizeForCreation = 200;
        conf.anng.insertionRadiusCoefficient = 1.1;
        conf.anng.outgoingEdge = 40;
        conf.anng.incomingEdge = 100;

        // WEAVES on Audio
        conf.onng.numOfOutgoingEdges = 40;
        conf.onng.numOfIncomingEdges = 100;

        conf.anns_k = 100;
        conf.anns_repeat = 10;
        conf.epsilons = {0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20};
        conf.explore_k = 1000;
    }

    return conf;
}

struct GraphPaths {
    std::filesystem::path ngt_dir;

    GraphPaths(const Dataset& ds) : ngt_dir(ds.data_root() / ds.name() / "ngt") {}

    std::string anng_base_name(const ANNGParams& p) const {
        return string_format("anng_K%u_E%u_out%u_in%u_eps%.2f",
                             p.edgeSizeForCreation,
                             p.edgeSizeLimitForCreation,
                             p.outgoingEdge,
                             p.incomingEdge,
                             p.insertionRadiusCoefficient);
    }

    std::string onng_base_name(const ANNGParams& anng, const ONNGParams& onng) const {
        return string_format("onng_out%u_in%u_%s", onng.numOfOutgoingEdges, onng.numOfIncomingEdges, anng_base_name(anng).c_str());
    }

    std::string graph_directory() const { return ngt_dir.string(); }

    std::string anng_index_path(const ANNGParams& p) const { return (ngt_dir / anng_base_name(p)).string(); }

    std::string onng_index_path(const ANNGParams& anng, const ONNGParams& onng) const {
        return (ngt_dir / onng_base_name(anng, onng)).string();
    }

    std::string log_file(const ANNGParams& anng, const ONNGParams& onng) const {
        return (ngt_dir / (onng_base_name(anng, onng) + ".log")).string();
    }
};

static void run_create_anng(const Dataset& ds, const ANNGParams& params, const std::string& indexPath, const LoadedData& base_data) {
    if (std::filesystem::exists(indexPath)) {
        log("ANNG Index already exists at %s\n", indexPath.c_str());
        return;
    }

    log("Creating ANNG Index at %s\n", indexPath.c_str());

    NGT::Property property;
    property.indexType = NGT::Index::Property::IndexType::GraphAndTree;
    property.threadPoolSize = 1;
    property.batchSizeForCreation = params.batchSizeForCreation;
    property.graphType = NGT::Property::GraphType::GraphTypeANNG;
    property.objectType = NGT::ObjectSpace::ObjectType::Float;
    property.distanceType = NGT::Index::Property::DistanceType::DistanceTypeL2;
    property.dimension = base_data.dim;
    property.edgeSizeForCreation = params.edgeSizeForCreation;
    property.edgeSizeLimitForCreation = params.edgeSizeLimitForCreation;
    property.outgoingEdge = params.outgoingEdge;
    property.incomingEdge = params.incomingEdge;
    property.edgeSizeForSearch = 0;
    property.insertionRadiusCoefficient = params.insertionRadiusCoefficient;
    property.pathAdjustmentInterval = 0;
    property.dynamicEdgeSizeBase = 30;
    log("\n--------------------------------------------------------------------------------\n");
    log("Creating ANNG Index: %s\n", indexPath.c_str());
    log("  Properties:\n");
    log("    EdgeSizeForCreation: %d\n", property.edgeSizeForCreation);
    log("    EdgeSizeLimitForCreation: %d\n", property.edgeSizeLimitForCreation);
    log("    InsertionRadiusCoefficient: %.2f\n", property.insertionRadiusCoefficient);
    log("    OutgoingEdge: %d\n", property.outgoingEdge);
    log("    IncomingEdge: %d\n", property.incomingEdge);
    log("    ThreadPoolSize: %d\n", property.threadPoolSize);

    log("Start creating ANNG index files...\n");
    NGT::Index::create(indexPath, property);
    NGT::Index index(indexPath);

    log("Appending %u vectors...\n", base_data.num);
    size_t dim = base_data.dim;
    for (size_t i = 0; i < base_data.num; i++) {
        std::vector<float> obj(base_data.data + i * dim, base_data.data + (i + 1) * dim);
        index.append(obj);
    }

    log("Building ANNG Index (Threads: %d)...\n", property.threadPoolSize);
    StopW sw;
    index.createIndex(property.threadPoolSize);
    log("ANNG Build Time: %.2f s\n", sw.getElapsedTimeMicro() / 1000000.0);

    index.save();
    index.close();
}

static void run_create_onng(const std::string& anngPath, const std::string& onngPath, const ONNGParams& params) {
    if (std::filesystem::exists(onngPath)) {
        log("ONNG Index already exists at %s\n", onngPath.c_str());
        return;
    }

    log("\n--------------------------------------------------------------------------------\n");
    log("Converting ANNG to ONNG: %s -> %s\n", anngPath.c_str(), onngPath.c_str());

    NGT::GraphOptimizer graphOptimizer(false);
    graphOptimizer.shortcutReduction = true;                        // (default=true)
    graphOptimizer.searchParameterOptimization = true;              // (default=true)
    graphOptimizer.prefetchParameterOptimization = true;            // (default=true)
    graphOptimizer.accuracyTableGeneration = false;                 // does not work (default=true)
    graphOptimizer.gtEpsilon = params.gtEpsilon;                    // eps for searching the ground truth data (default=0.1)
    graphOptimizer.margin = params.margin;                          // (default=0.2)
    graphOptimizer.minNumOfEdges = params.minNumOfEdges;            // E (default=0)
    graphOptimizer.numOfQueries = params.numOfQueries;              // # of ground truth objects  (default=100)
    graphOptimizer.numOfResults = params.numOfResults;              // # of resultant objects (default=20)
    graphOptimizer.numOfOutgoingEdges = params.numOfOutgoingEdges;  // i (ONNG parameter) (default=10)
    graphOptimizer.numOfIncomingEdges = params.numOfIncomingEdges;  // o (ONNG parameter) (default=120)

    log("  Optimizer Settings:\n");
    log("    ShortcutReduction: %s\n", graphOptimizer.shortcutReduction ? "true" : "false");
    log("    SearchParamOptimization: %s\n", graphOptimizer.searchParameterOptimization ? "true" : "false");
    log("    PrefetchParamOptimization: %s\n", graphOptimizer.prefetchParameterOptimization ? "true" : "false");
    log("    AccuracyTableGeneration: %s\n", graphOptimizer.accuracyTableGeneration ? "true" : "false");
    log("    GTEpsilon: %.2f\n", graphOptimizer.gtEpsilon);
    log("    Margin: %.2f\n", graphOptimizer.margin);
    log("    MinNumOfEdges: %d\n", graphOptimizer.minNumOfEdges);
    log("    NumOfQueries: %d\n", graphOptimizer.numOfQueries);
    log("    NumOfResults: %d\n", graphOptimizer.numOfResults);
    log("    NumOfOutgoingEdges: %d\n", graphOptimizer.numOfOutgoingEdges);
    log("    NumOfIncomingEdges: %d\n", graphOptimizer.numOfIncomingEdges);

    StopW sw;
    graphOptimizer.execute(anngPath, onngPath);
    log("ONNG Optimization Time: %.2f s\n", sw.getElapsedTimeMicro() / 1000000.0);
}

static void run_exploration_test_suite(NGT::Index& index, const Dataset& ds, const DatasetConfig& conf, const LoadedData& query_data) {
    std::string entry_file = ds.explore_entry_vertex_file();
    std::string explore_gt_file = ds.explore_groundtruth_file(false);

    if (!std::filesystem::exists(entry_file) || !std::filesystem::exists(explore_gt_file)) {
        log("Skipping exploration test: missing files.\n");
        return;
    }

    log("Loading exploration data...\n");
    auto explore_queries = ds.load_explore_query();

    // Load entry nodes
    size_t dim, num;
    auto entry_ptr = ivecs_read(entry_file.c_str(), dim, num);
    if (!entry_ptr) {
        log("Failed to load entry nodes\n");
        return;
    }
    std::vector<uint32_t> entry_nodes;
    entry_nodes.reserve(num);
    for (size_t i = 0; i < num; ++i) {
        entry_nodes.push_back(entry_ptr[i * dim]);  // Assuming 1st element is the entry node
    }

    // Load exploration GT
    size_t gt_dim, gt_num;
    auto gt_ptr = ivecs_read(explore_gt_file.c_str(), gt_dim, gt_num);
    std::vector<std::vector<uint32_t>> explore_gt(gt_num);
    for (size_t i = 0; i < gt_num; ++i) {
        explore_gt[i].reserve(conf.explore_k);
        for (size_t j = 0; j < conf.explore_k && j < gt_dim; ++j) {
            explore_gt[i].push_back(gt_ptr[i * gt_dim + j]);
        }
        std::sort(explore_gt[i].begin(), explore_gt[i].end());
    }

    log("Running Exploration Test (k=%d, eps=%.3f)...\n", conf.explore_k, conf.explore_epsilon);
    test_ngt_explore(index,
                     explore_queries.data,
                     explore_queries.num,
                     explore_queries.dim,
                     explore_gt,
                     entry_nodes,
                     conf.explore_k,
                     conf.explore_epsilon);
}

static void run_test(const Dataset& ds, const DatasetConfig& conf, const GraphPaths& paths) {
    std::string anng_path = paths.anng_index_path(conf.anng);
    std::string onng_path = paths.onng_index_path(conf.anng, conf.onng);
    std::string log_path = paths.log_file(conf.anng, conf.onng);

    std::filesystem::create_directories(paths.graph_directory());

    if (std::filesystem::exists(log_path)) {
        log("Log file exists, skipping: %s\n", log_path.c_str());
        return;
    }

    set_log_file(log_path, true);
    attach_cerr_to_log();
    attach_cout_to_log();

    log("=== NGT Benchmark %s ===\n", ds.name());

    // Memory usage before build
    size_t rss_before = getProcessCurrentRSS();
    log("Initial Memory Usage: %.2f MB\n", rss_before / (1024.0 * 1024.0));

    size_t mem_before_load = getProcessCurrentRSS();
    auto base_data = ds.load_base();
    size_t mem_after_load = getProcessCurrentRSS();
    log("Base data memory usage: %.2f MB\n", (mem_after_load - mem_before_load) / (1024.0 * 1024.0));

    StopW build_timer;

    // 1. ANNG
    run_create_anng(ds, conf.anng, anng_path, base_data);

    // 2. ONNG
    run_create_onng(anng_path, onng_path, conf.onng);

    double total_build_time = build_timer.getElapsedTimeMicro() / 1000000.0;
    log("Total Graph Construction Time: %.2f s\n", total_build_time);

    // 3. Test
    log("Loading ONNG for testing: %s\n", onng_path.c_str());
    NGT::Index index(onng_path);

    // Compute Stats
    // Assuming base groundtruth is the top_list file used in l2-float_stats
    std::string top_list_file = ds.base_groundtruth_file(false);
    if (std::filesystem::exists(top_list_file)) {
        statistics::compute_stats(&index, top_list_file.c_str());
    } else {
        log("Skipping stats: TopList file not found at %s\n", top_list_file.c_str());
    }

    auto query_data = ds.load_query();
    auto ground_truth = ds.load_groundtruth(conf.anns_k);

    log("\n--------------------------------------------------------------------------------\n");
    log("--- ANNS Test ---\n");
    test_ngt_anns(index, query_data.data, query_data.num, query_data.dim, ground_truth, conf.anns_k, conf.epsilons, conf.anns_repeat);

    log("\n--------------------------------------------------------------------------------\n");
    log("--- Exploration Test ---\n");
    run_exploration_test_suite(index, ds, conf, query_data);

    reset_log_to_console();
}

int main(int argc, char** argv) {
    log("NGT Benchmark Suite\n");

    std::string data_root = DATA_PATH;
    DatasetName ds_name = DatasetName::ENRON;
    bool do_run = true;

    if (argc > 1) {
        DatasetName params_ds = DatasetName::from_string(argv[1]);
        if (params_ds.is_valid()) ds_name = params_ds;

        if (argc > 2) data_root = argv[2];
    }

    if (ds_name == DatasetName::ALL) {
        for (const auto& name : DatasetName::all()) {
            Dataset ds(name, data_root);
            auto conf = get_dataset_config(name);
            GraphPaths paths(ds);
            run_test(ds, conf, paths);
        }
    } else {
        Dataset ds(ds_name, data_root);
        auto conf = get_dataset_config(ds_name);
        GraphPaths paths(ds);
        run_test(ds, conf, paths);
    }

    return 0;
}
