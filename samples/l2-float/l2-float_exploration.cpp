
#include	"NGT/Index.h"
#include	"NGT/GraphOptimizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <assert.h>


#include <vector>
#include <unordered_set>

/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference
 *https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/
auto fvecs_read(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::cerr << "error when accessing test file, size is: " << file_size << " message: " << ec.message() << std::endl;
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
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

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const uint32_t ground_truth_dims, const size_t k)
{
    auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

int main(int argc, char **argv)
{
  
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
 

  auto indexPath        = R"(c:/Data/Feature/SIFT1M/NGT/anng500-onng50_150_default)";
  auto objectFile       = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  auto queryFile       = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_explore_query.fvecs)";
  auto groundtruthFile = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_explore_ground_truth.ivecs)";
  auto entryNodeFile  = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_explore_entry_node.ivecs)";
  uint32_t maxK = 1000;
  unsigned seed = 161803398;
  srand(seed);


  auto index = NGT::Index(indexPath);
  NGT::Property	property;
  index.getProperty(property);

  std::cout << "dimension: " << property.dimension << std::endl;
  std::cout << "edgeSizeForCreation: " << property.edgeSizeForCreation << std::endl;
  std::cout << "threadPoolSize: " << property.threadPoolSize << std::endl;
  std::cout << "objectType: " << property.objectType << std::endl;                    // Uint8		= 1, Float		= 2
  std::cout << "distanceType: " << property.distanceType << std::endl;                // DistanceTypeL2			= 1,
  std::cout << "databaseType: " << property.databaseType << std::endl;                // Memory			= 1,
  std::cout << "graphType: " << property.graphType << std::endl;                      // GraphTypeANNG	= 1
  std::cout << "indexType: " << property.indexType << std::endl;                      // GraphAndTree		= 1,
  std::cout << "accuracyTable: " << property.accuracyTable << std::endl;               


  // query data
  size_t query_num, query_dim;
  auto query_data = fvecs_read(queryFile, query_dim, query_num);

  // query ground truth
  size_t groundtruth_num, groundtruth_dim;
  auto groundtruth_f = fvecs_read(groundtruthFile, groundtruth_dim, groundtruth_num);
  const auto ground_truth = (uint32_t*)groundtruth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)

  // entry node ids
  size_t entrynode_num, entrynode_dim;
  auto entrynode_f = fvecs_read(entryNodeFile, entrynode_dim, entrynode_num);
  const auto entry_node = (uint32_t*)entrynode_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)

  auto steps = 30;
  for (size_t i = 3; i <= steps; i++) {
    const auto K = maxK;
    const auto max_distance_count = uint32_t(K + (K/1 * i));

    const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, K);
    auto time_begin = std::chrono::steady_clock::now();

    size_t correct = 0;
    for (unsigned i = 0; i < query_num; i++) {
      auto entry_node_index = (uint32_t)  (entry_node[i * entrynode_dim] + 1);
      auto query = std::vector(query_data.get() + i * query_dim, query_data.get() + i * query_dim + query_dim);
      NGT::SearchQuery		sc(query);
      NGT::ObjectDistances	objects;
      sc.setResults(&objects);
      sc.setSize(K);
      sc.setEpsilon(0.00f); // why does -0.005f improve the quality?

      index.explore(sc, entry_node_index, max_distance_count);

      // compare answer with ann
      auto answer = answers[i];
      for (size_t r = 0; r < K; r++) 
        if (answer.find(objects[r].id - 1) != answer.end()) correct++; // all ids in the index to high by 1 value
    }

    auto time_end = std::chrono::steady_clock::now();
    auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / query_num;
    auto recall = 1.0f * correct / (query_num * K);
    std::cout << "k and p " << K << ", max_distance_count " << max_distance_count << ", recall " << recall << " time_us_per_query " << time_us_per_query << std::endl;
  }

  return 0;
}