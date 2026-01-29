
#include <assert.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "NGT/GraphOptimizer.h"
#include "NGT/Index.h"


/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference
 *https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/
auto fvecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::cerr << "error when accessing test file, size is: " << file_size << " message: " << ec.message() << std::endl;
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open()) {
        std::cerr << "could not open " << fname << std::endl;
        abort();
    }

    int dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert(file_size % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = file_size / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<float[]>(n * (dims + 1));
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(float));
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(float));

    ifstream.close();
    return x;
}

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth,
                                                                  const size_t ground_truth_size,
                                                                  const uint32_t ground_truth_dims,
                                                                  const size_t k) {
    auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++) {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

template <typename... Args>
std::string string_format(const char* fmt, Args... args) {
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
}

int main(int argc, char** argv) {
#if defined(__AVX2__)
    std::cout << "use AVX2  ..." << std::endl;
#elif defined(__AVX__)
    std::cout << "use AVX  ..." << std::endl;
#else
    std::cout << "use arch  ..." << std::endl;
#endif

#if defined(NGT_AVX2)
    std::cout << "use NGT_AVX2  ..." << std::endl;
#elif defined(NGT_NO_AVX)
    std::cout << "use NGT_NO_AVX  ..." << std::endl;
#endif

    auto readOnly = false;
    uint32_t k = 1000;
    unsigned seed = 161803398;
    srand(seed);

    // // ----------------------------------------- SIFT1M ------------------------------------------------
    // auto indexPath       = R"(e:/Data/Feature/SIFT1M/NGT/onng in30 out110 noTable (anng K200 eps1.1))";
    // auto queryFile       = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_explore_query.fvecs)";
    // auto groundtruthFile = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_explore_ground_truth.ivecs)";
    // auto entryNodeFile   = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_explore_entry_vertex.ivecs)";

    // // ----------------------------------------- Glove ------------------------------------------------
    // auto indexPath       = R"(e:/Data/Feature/GloVe/NGT/onng in15 out155 noTable (anng K200 eps1.1))";
    // auto queryFile       = R"(e:/Data/Feature/GloVe/glove-100/glove-100_explore_query.fvecs)";
    // auto groundtruthFile = R"(e:/Data/Feature/GloVe/glove-100/glove-100_explore_ground_truth.ivecs)";
    // auto entryNodeFile   = R"(e:/Data/Feature/GloVe/glove-100/glove-100_explore_entry_vertex.ivecs)";
    // auto eps             = 0.03f;

    // ----------------------------------------- Enron ------------------------------------------------
    // auto indexPath       = R"(e:/Data/Feature/Enron/NGT/onng in20 out100 noTable (anng K200 eps1.1))";
    // auto queryFile       = R"(e:/Data/Feature/Enron/enron/enron_explore_query.fvecs)";
    // auto groundtruthFile = R"(e:/Data/Feature/Enron/enron/enron_explore_ground_truth.ivecs)";
    // auto entryNodeFile   = R"(e:/Data/Feature/Enron/enron/enron_explore_entry_vertex.ivecs)";
    // auto eps             = 0.012f;

    // ----------------------------------------- Audio ------------------------------------------------
    // auto indexPath       = R"(e:/Data/Feature/Audio/NGT/onng in40 out100 noTable (anng K200 eps1.1))";
    // auto queryFile       = R"(e:/Data/Feature/Audio/audio/audio_explore_query.fvecs)";
    // auto groundtruthFile = R"(e:/Data/Feature/Audio/audio/audio_explore_ground_truth.ivecs)";
    // auto entryNodeFile   = R"(e:/Data/Feature/Audio/audio/audio_explore_entry_vertex.ivecs)";
    // auto eps             = 0.03f;

    // ----------------------------------------- Deep1M ------------------------------------------------
    auto indexPath = R"(e:/Data/Feature/Deep1M/NGT/in30 out110 noTable (anng K200 eps1.1))";
    auto queryFile = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_explore_query.fvecs)";
    auto groundtruthFile = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_explore_ground_truth.ivecs)";
    auto entryNodeFile = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_explore_entry_vertex.ivecs)";
    auto eps = 0.03f;

    std::cout << "Load index (readOnly=" << readOnly << ")" << std::endl;
    auto index = NGT::Index(indexPath, readOnly);
    NGT::Property property;
    index.getProperty(property);

    std::cout << "dimension: " << property.dimension << std::endl;
    std::cout << "edgeSizeForCreation: " << property.edgeSizeForCreation << std::endl;
    std::cout << "threadPoolSize: " << property.threadPoolSize << std::endl;
    std::cout << "objectType: " << property.objectType << std::endl;      // Uint8		= 1, Float		= 2
    std::cout << "distanceType: " << property.distanceType << std::endl;  // DistanceTypeL2			= 1,
    std::cout << "databaseType: " << property.databaseType << std::endl;  // Memory			= 1,
    std::cout << "graphType: " << property.graphType << std::endl;        // GraphTypeANNG	= 1
    std::cout << "indexType: " << property.indexType << std::endl;        // GraphAndTree		= 1,
    std::cout << "accuracyTable: " << property.accuracyTable << std::endl;
    std::cout << "eps: " << eps << std::endl;

    // query data
    size_t query_num, query_dim;
    auto query_data = fvecs_read(queryFile, query_dim, query_num);

    // query ground truth
    size_t groundtruth_num, groundtruth_dim;
    auto groundtruth_f = fvecs_read(groundtruthFile, groundtruth_dim, groundtruth_num);
    const auto ground_truth = (uint32_t*)groundtruth_f.get();  // not very clean, works as long as sizeof(int) == sizeof(float)
    const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, k);

    // entry node ids
    size_t entrynode_num, entrynode_dim;
    auto entrynode_f = fvecs_read(entryNodeFile, entrynode_dim, entrynode_num);
    const auto entry_node = (uint32_t*)entrynode_f.get();  // not very clean, works as long as sizeof(int) == sizeof(float)

    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= 3; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i - 1)) : (k_factor * i));

            auto time_begin = std::chrono::steady_clock::now();

            size_t correct = 0;
            size_t empty_list = 0;
            size_t short_list = 0;
            size_t max_ids_in_result = 0;
            for (unsigned q = 0; q < query_num; q++) {
                auto entry_node_index = (uint32_t)(entry_node[q * entrynode_dim] + 1);
                auto query = std::vector(query_data.get() + q * query_dim, query_data.get() + q * query_dim + query_dim);
                NGT::SearchQuery sc(query);
                NGT::ObjectDistances objects;
                sc.setResults(&objects);
                sc.setSize(k);
                sc.setEpsilon(eps);  // why does -0.005f improve the quality?

                index.explore(sc, entry_node_index, max_distance_count);

                // compare answer with ann
                auto answer = answers[q];
                for (size_t r = 0; r < k; r++)
                    if (answer.find(objects[r].id - 1) != answer.end()) correct++;  // all ids in the index to high by 1 value

                max_ids_in_result += k - objects.size();
                if (objects.size() < k) short_list++;
                if (objects.size() == 0) empty_list++;
            }

            auto time_end = std::chrono::steady_clock::now();
            auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / query_num;
            auto recall = 1.0f * correct / (query_num * k);
            std::cout << string_format(
                "k and p %5d, max_distance_count %6d, recall %.4f, time_us_per_query %6d, empty_list %6d, short_list %6d, "
                "max_ids_in_result %6d\n",
                k,
                max_distance_count,
                recall,
                time_us_per_query,
                empty_list,
                short_list,
                max_ids_in_result);
        }
    }

    return 0;
}