
#include	"NGT/Index.h"
#include	"NGT/GraphOptimizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
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

template<typename... Args>
std::string string_format(const char* fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
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
 
  auto readOnly         = true;
  auto treeDisabled     = false; // true if ONNG was build with PANNG 

  unsigned K = 100;  
  unsigned repeat_test = 1;
  unsigned seed = 161803398;
  srand(seed);

  // SIFT
  // auto queryFile        = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_query.fvecs)";
  // auto groundtruthFile	= R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_groundtruth.ivecs)";
  // auto indexPath        = R"(e:/Data/Feature/SIFT1M/NGT/in30 out110 noTable (anng200 eps1.1))";
  // std::vector<float> exploration_coefficients = { -0.03f, -0.02f, -0.01f, -0.005f, -0.001f, 0.005f, 0.01f, 0.03f, 0.05f};

  // // GloVe
  // auto queryFile        = R"(e:/Data/Feature/GloVe/glove-100/glove-100_query.fvecs)";
  // auto groundtruthFile  = R"(e:/Data/Feature/GloVe/glove-100/glove-100_groundtruth.ivecs)";
  // auto indexPath        = R"(e:/Data/Feature/GloVe/NGT/onng in15 out155 noTable (anng K200 eps1.1))";
  // std::vector<float> exploration_coefficients = { 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.1f }; 

  // Enron
  // auto queryFile        = R"(e:/Data/Feature/Enron/enron/enron_query.fvecs)";
  // auto groundtruthFile  = R"(e:/Data/Feature/Enron/enron/enron_groundtruth_top1000.ivecs)";
  // auto indexPath        = R"(e:/Data/Feature/Enron/NGT/onng in20 out100 noTable (anng K200 eps1.1))";
  // // std::vector<float> exploration_coefficients = {0.015f, 0.016f, 0.017f, 0.018f, 0.02f, 0.03f, 0.05f}; // TOP20
  // std::vector<float> exploration_coefficients = {0.01f, 0.011f, 0.012f, 0.013f, 0.014f, 0.015f, 0.018f, 0.03f};
  // // std::vector<float> exploration_coefficients = {0.015f, 0.016f, 0.017f, 0.018f, 0.019f, 0.02f, 0.021f, 0.022f, 0.023f, 0.024f, 0.025f, 0.026f, 0.027f, 0.028f, 0.029f, 0.03f, 0.05f};
  // K = 100;
  // repeat_test = 20;

  // // Audio
  // auto queryFile        = R"(e:/Data/Feature/Audio/audio/audio_query.fvecs)";
  // auto groundtruthFile  = R"(e:/Data/Feature/Audio/audio/audio_groundtruth_top1000.ivecs)";
  // auto indexPath        = R"(e:/Data/Feature/Audio/NGT/onng in40 out100 noTable (anng K200 eps1.1))";
  // std::vector<float> exploration_coefficients = {0.00f, 0.01f, 0.02f, 0.03f, 0.035f, 0.04f, 0.05f, 0.07f};
  // K = 100;
  // repeat_test = 50;


  // Deep1M
  auto queryFile        = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_query.fvecs)";
  auto groundtruthFile  = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_groundtruth.ivecs)";
  auto indexPath        = R"(e:/Data/Feature/Deep1M/NGT/in30 out110 noTable (anng K200 eps1.1))";
  std::vector<float> exploration_coefficients = {0.00f, 0.01f, 0.02f, 0.03f, 0.035f, 0.04f, 0.05f, 0.07f};
  K = 100;
  repeat_test = 1;

  // convert to txt files
  // auto objectFile       = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  // auto objectFile_text  = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.txt)";
  // size_t count, dim;
  // auto data = fvecs_read(objectFile, dim, count);

  // auto out = std::ofstream(objectFile_text, std::ios::out);
  // for (size_t i = 0; i < count; i++) {
  //   int offset = i * dim; 

  //   for (size_t d = 0; d < dim; d++) {
  //     out << (int32_t) data[offset + d];

  //     if(d+1 < dim)
  //       out << " ";
  //   }
  //   out << std::endl;   
  // }
  // out.close();






  std::cout << "Load index (readOnly=" << readOnly << ")" << std::endl;
  auto index = NGT::Index(indexPath, readOnly);
  NGT::Property	property;
  index.getProperty(property);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading index" << std::endl;


  // auto new_vector = std::vector<float>(128);
  // std::fill(new_vector.begin(),new_vector.end(), 1);
  // const float* new_data = new_vector.data();
  // index.append(new_data, 128);
  // index.remove(1, true);

  std::cout << "dimension: " << property.dimension << std::endl;
  std::cout << "edgeSizeForCreation: " << property.edgeSizeForCreation << std::endl;
  std::cout << "edgeSizeLimitForCreation: " << property.edgeSizeLimitForCreation << std::endl;
  std::cout << "edgeSizeForSearch: " << property.edgeSizeForSearch << std::endl;
  std::cout << "insertionRadiusCoefficient: " << property.insertionRadiusCoefficient << std::endl;
  std::cout << "threadPoolSize: " << property.threadPoolSize << std::endl;
  std::cout << "outgoingEdge: " << property.outgoingEdge << std::endl;
  std::cout << "incomingEdge: " << property.incomingEdge << std::endl;
  std::cout << "batchSizeForCreation: " << property.batchSizeForCreation << std::endl;
  std::cout << "threadPoolSize: " << property.threadPoolSize << std::endl;
  std::cout << "pathAdjustmentInterval: " << property.pathAdjustmentInterval << std::endl;
  std::cout << "dynamicEdgeSizeBase: " << property.dynamicEdgeSizeBase << std::endl;
  std::cout << "buildTimeLimit: " << property.buildTimeLimit << std::endl;
  std::cout << "objectType: " << property.objectType << std::endl;                    // Uint8		= 1, Float		= 2
  std::cout << "distanceType: " << property.distanceType << std::endl;                // DistanceTypeL2			= 1,
  std::cout << "databaseType: " << property.databaseType << std::endl;                // Memory			= 1,
  std::cout << "graphType: " << property.graphType << std::endl;                      // GraphTypeANNG	= 1
  std::cout << "indexType: " << property.indexType << std::endl;                      // GraphAndTree		= 1,
  std::cout << "accuracyTable: " << property.accuracyTable << std::endl;               


  // query data
  std::cout << "Load query data" << std::endl;
  size_t query_num, query_dim;
  auto query_data = fvecs_read(queryFile, query_dim, query_num);

  // query ground truth
  size_t groundtruth_num, groundtruth_dim;
  auto groundtruth_f = fvecs_read(groundtruthFile, groundtruth_dim, groundtruth_num);
  const auto ground_truth = (uint32_t*)groundtruth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
  const auto answers = get_ground_truth(ground_truth, groundtruth_num, groundtruth_dim, K);
  std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading query data" << std::endl;


  std::cout << "Evaluate graph with TOP" << K << std::endl;
  for (float exploration_coefficient : exploration_coefficients) {


    auto time_begin = std::chrono::steady_clock::now();

    size_t correct = 0;    
    for (unsigned t = 0; t < repeat_test; t++) {
      for (unsigned i = 0; i < query_num; i++) {
        auto query = std::vector(query_data.get() + i * query_dim, query_data.get() + i * query_dim + query_dim);
        NGT::SearchQuery		sc(query);
        NGT::ObjectDistances	objects;
        sc.setResults(&objects);
        sc.setSize(K);
        sc.setEpsilon(exploration_coefficient);
        //sc.setExpectedAccuracy(0.7f); // needs accuracy table in the graph file

        if(treeDisabled)
          index.searchUsingOnlyGraph(sc);
        else
          index.search(sc);

        // compare answer with ann
        auto answer = answers[i];
        for (size_t r = 0; r < K; r++)
          if (answer.find(objects[r].id - 1) != answer.end()) correct++; // all ids in the index to high by 1 value
      }
    }

    auto time_end = std::chrono::steady_clock::now();
    auto time_us_per_query = (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / (query_num * repeat_test);
    auto recall = 1.0f * correct / repeat_test / (query_num * K);
    std::cout << string_format("exploration_coefficient %.3f, recall %.4f, time_us_per_query %8d \n", exploration_coefficient, recall, time_us_per_query);
    // if (recall > 1.0)
    //   break;
  }

  return 0;
}