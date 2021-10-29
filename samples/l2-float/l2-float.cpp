
#include	"NGT/Index.h"
#include	"NGT/GraphOptimizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <assert.h>

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

// index construction
int createANNGIndex(const char * indexPath, const char * featurePath, const int edgeSizeForCreation, const int threadNumber) {
  try {
    size_t dims;
    size_t count;
    auto features = fvecs_read(featurePath, dims, count);

    // ngt create -i t -g a -S 0 -e 0.1 -E 100 -d 128 -o c -D 2 anng-index vector-data.dat
    // indexType is createGraphAndTree
    // graphType is GraphTypeANNG
    // edgeSizeForSearch is 0
    // insertionRadiusCoefficient is 1.1f
    // edgeSizeForCreation is 100
    // dimensions is 128
    // objectType is Uint8
    // distanceType is DistanceTypeL2
    // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Command.cpp#L39
    NGT::Property	property;
    property.edgeSizeForCreation = edgeSizeForCreation;
    property.dimension		= (int)dims;
    property.objectType		= NGT::ObjectSpace::ObjectType::Float;
    property.distanceType	= NGT::Index::Property::DistanceType::DistanceTypeL2;
    property.threadPoolSize = threadNumber;
    property.insertionRadiusCoefficient = 1.03;

    // WEAVES verwendet undirected eges 
    // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
    // https://github.com/Lsyhprum/WEAVESS/blob/master/test/main.cpp#L233
    // https://github.com/Lsyhprum/WEAVESS/blob/master/src/component_init.cpp#L1287
    // property.unknown = 200;             // K is nn (ANNG parameter)
    // property.edgeSizeForCreation = 230; // L is ef_construction (ANNG parameter)
    // property.outgoingEdge = 30;         // out (ONNG parameter)
    // property.incomingEdge = 100;        // in (ONNG parameter)

    std::cout << "Start creating ANNG index files" << std::endl;
    NGT::Index::create(indexPath, property);
    NGT::Index	index(indexPath);
    
    std::cout << "Copy features vectores to index" << std::endl;
    for (size_t i = 0; i < count; i++) {
      auto feature_array = &features[i * dims];
      auto feature = std::vector<float>(feature_array, feature_array + dims);
      index.append(std::move(feature));
    }

    std::cout << "Create ANNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();
      index.enableLog();
      index.createIndex(threadNumber);
      auto time_end = std::chrono::steady_clock::now();
      auto time_per_seconds = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count();
      std::cout << "time_per_seconds " << time_per_seconds << std::endl;
    }

    std::cout << "Save ANNG" << std::endl;
    index.save();

    index.close();
  } catch (NGT::Exception &err) {
    std::cerr << "Error " << err.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Error" << std::endl;
    return 1;
  }

  return 0;
}


// index construction
int refineANNGIndex(const char * anngIndexPath, const char * rnngIndexPath, const int edgeSizeForCreation, const int threadNumber) {
  try {

    std::cout << "Load ANNG index files" << std::endl;
    auto index = NGT::Index(anngIndexPath);

    // Which properties need to be set are copied from here https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/onng_ngt.py
    // Their values are copied from here https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
    // And how to use them here https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Command.cpp
    NGT::Property	property;
    index.getProperty(property);
    property.edgeSizeForCreation = edgeSizeForCreation; 
    property.objectAlignment = NGT::Index::Property::ObjectAlignment::ObjectAlignmentTrue;
    property.threadPoolSize = threadNumber;
    property.outgoingEdge = 10;
    property.outgoingEdge = 120;
    property.dynamicEdgeSizeBase = 10;
    index.setProperty(property);


    // https://arxiv.org/pdf/1810.07355.pdf
    // ANNG edgeSizeForCreation=200 insertionRadiusCoefficient=0.1

    std::cout << "Refine ANNG to RNNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();
      NGT::GraphReconstructor::refineANNG(index, true);
      auto time_end = std::chrono::steady_clock::now();
      auto time_per_seconds = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count();
      std::cout << "time_per_seconds " << time_per_seconds << ", GraphType" << property.graphType << std::endl;
    }
    
    std::cout << "Save RNNG" << std::endl;
    index.save(rnngIndexPath);

    index.close();
  } catch (NGT::Exception &err) {
    std::cerr << "Error " << err.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Error" << std::endl;
    return 1;
  }

  return 0;
}

int reconstructONNGIndex(const char * anngIndexPath, const char * onngIndexPath) {
  try {

    // ngt reconstruct-graph -m S -o outdegree -i indegree anng-index onng-index
    // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Command.cpp#L700
    std::cout << "Convert ANNG to ONNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();

      NGT::GraphOptimizer graphOptimizer(false);
      graphOptimizer.shortcutReduction = true;
      graphOptimizer.searchParameterOptimization = true;
      graphOptimizer.prefetchParameterOptimization = true;
      graphOptimizer.accuracyTableGeneration = false;
      graphOptimizer.minNumOfEdges =  0;
      graphOptimizer.gtEpsilon = 0.1;    // eps for searching the ground truth data
      graphOptimizer.margin = 0.3;       // 

      graphOptimizer.numOfQueries = 100;  // # of ground truth objects
      graphOptimizer.numOfResults = 100;  // # of resultant objects

      graphOptimizer.numOfOutgoingEdges = 50;
      graphOptimizer.numOfIncomingEdges = 150;

      graphOptimizer.execute(std::string(anngIndexPath), std::string(onngIndexPath));
      

      auto time_end = std::chrono::steady_clock::now();
      auto time_per_seconds = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count();
      std::cout << "time_per_seconds " << time_per_seconds << std::endl;
    }

  } catch (NGT::Exception &err) {
    std::cerr << "Error " << err.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Error" << std::endl;
    return 1;
  }

  return 0;
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
 

  auto anngIndexPath    = R"(c:/Data/Feature/SIFT1M/NGT/anng500)";
  auto rnngIndexPath    = R"(c:/Data/Feature/SIFT1M/NGT/rnng500)";
  auto onngIndexPath    = R"(c:/Data/Feature/SIFT1M/NGT/anng500-onng50_150_default-redo)";
  auto objectFile       = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  auto queryFile        = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_query.fvecs)";
  auto groundtruthFile	= R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_groundtruth.ivecs)";
  
  auto edgeSizeForCreation = 300;
  auto threadNumber = 1;

  // anng index construction
  /*int constructState = createANNGIndex(anngIndexPath, objectFile, edgeSizeForCreation, threadNumber);
  if(constructState == 1)
    return 1;*/

  // anng index refinement
  /*int refineState = refineANNGIndex(anngIndexPath, rnngIndexPath, edgeSizeForCreation, threadNumber);
  if(refineState == 1)
    return 1;*/

  int reconstructState = reconstructONNGIndex(anngIndexPath, onngIndexPath);
  if(reconstructState == 1)
    return 1;
    
  return 0;
}


