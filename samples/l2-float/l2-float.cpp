
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
int createANNGIndex(const char * indexPath, const char * featurePath, const int threadNumber) {
  try {
    size_t dims;
    size_t count;
    auto features = fvecs_read(featurePath, dims, count);
    
    // ONNG defaults
    // Which properties need to be set are copied from here 
    // https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/onng_ngt.py
    //
    // Their values are copied from here 
    // https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml#L378
    // https://github.com/yahoojapan/NGT/tree/main/bin/ngt
    //
    // And how to use them here 
    // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Command.cpp
    NGT::Property	property;
    property.indexType = NGT::Index::Property::IndexType::GraphAndTree;
    property.threadPoolSize = threadNumber;    // default=8
    property.batchSizeForCreation = 500;       // default=200
    property.graphType = NGT::Property::GraphType::GraphTypeANNG;
    property.objectType	= NGT::ObjectSpace::ObjectType::Float;
    property.distanceType	= NGT::Index::Property::DistanceType::DistanceTypeL2;
    property.dimension = (int) dims;
    property.edgeSizeForCreation = 100;         // default=10
    property.edgeSizeLimitForCreation = 100;    // default=10
    property.outgoingEdge = 10;                 // default=10
    property.incomingEdge = 120;                // default=80
    property.edgeSizeForSearch = 0;             // default=0
    property.insertionRadiusCoefficient = 1.1;  // default=1.1 (1.0 + epsilon)
    property.pathAdjustmentInterval = 0;        // default=0
    property.dynamicEdgeSizeBase = 30;          // default=30
    property.buildTimeLimit = 0;                // hours (default=0)
    
    // NGT repo SIFT1M
    // property.edgeSizeLimitForCreation = 100;     // same as edgeSizeForCreation
    // property.edgeSizeForCreation = 100;          // kc with kc > eo and kc > ei
    // property.insertionRadiusCoefficient = 1.1;   // eps_c´= 0.1
    // property.outgoingEdge = 10;                  // eo
    // property.incomingEdge = 120;                 // ei

    // quant NGT repo SIFT1M
    // https://github.com/yahoojapan/NGT/blob/main/bin/qbg/README.md
    // property.edgeSizeForCreation = 40;      // E=40
    // property.edgeSizeLimitForCreation = 40;
    // property.outgoingEdge = 64;
    // property.incomingEdge = 120; 

    // ONNG Paper Table 3 on SIFT1M
    // https://arxiv.org/pdf/1810.07355.pdf 
    // property.edgeSizeLimitForCreation = 200;     // same as edgeSizeForCreation
    // property.edgeSizeForCreation = 200;          // kc with kc > eo and kc > ei
    // property.insertionRadiusCoefficient = 1.1;   // eps_c´= 0.1
    // property.outgoingEdge = 30;                  // eo
    // property.incomingEdge = 110;                 // ei

    // WEAVES ANNG on SIFT1M 
    // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
    // https://github.com/Lsyhprum/WEAVESS/blob/master/test/main.cpp#L233
    // https://github.com/Lsyhprum/WEAVESS/blob/master/src/component_init.cpp#L1219
    // https://github.com/Lsyhprum/WEAVESS/blob/master/src/component_init.cpp#L1287
    // (uses undirected edges in their repository?) 
    // Index.h:searchForNNGInsertion(...) -> sc.size = NeighborhoodGraph::property.edgeSizeForCreation;
    // Index.cpp:insertMultipleSearchResults(...) -> size = neighborhoodGraph.NeighborhoodGraph::property.edgeSizeForCreation;
    // property.edgeSizeLimitForCreation = 200;             // K=200 is nn and limits the number of neighbors (ANNG parameter)
    // property.edgeSizeForCreation = 230;                  // L=230 is ef_construction and limits the number of search result (ANNG parameter)
    // property.insertionRadiusCoefficient = 1.03;          // not in weaves (ANNG parameter)
    // property.outgoingEdge = 30;                          // out=30 (ONNG parameter)
    // property.incomingEdge = 100;                         // in=100 (ONNG parameter)

    // // WEAVES ANNG on GloVe
    // property.edgeSizeLimitForCreation = 300;             // K=200 is nn and limits the number of neighbors (ANNG parameter)
    // property.edgeSizeForCreation = 300;                  // L=230 is ef_construction and limits the number of search result (ANNG parameter)
    // property.insertionRadiusCoefficient = 1.03;          // not in weaves (ANNG parameter)
    // property.outgoingEdge = 20;                          // out=30 (ONNG parameter)
    // property.incomingEdge = 100;                         // in=100 (ONNG parameter)

    // ONNG Paper Table 3 on GloVe
    // https://arxiv.org/pdf/1810.07355.pdf 
    // property.edgeSizeLimitForCreation = 200;     // same as edgeSizeForCreation
    // property.edgeSizeForCreation = 200;          // kc with kc > eo and kc > ei
    // property.insertionRadiusCoefficient = 1.1;   // eps_c´= 0.1
    // property.outgoingEdge = 15;                  // eo
    // property.incomingEdge = 155;                 // ei

    // WEAVES ANNG on Enron
    // property.edgeSizeLimitForCreation = 200;             // K=200 is nn and limits the number of neighbors (ANNG parameter)
    // property.edgeSizeForCreation = 200;                  // L=200 is ef_construction and limits the number of search result (ANNG parameter)
    // property.insertionRadiusCoefficient = 1.1;           // not in weaves but default (ANNG parameter)
    // property.outgoingEdge = 20;                          // (ONNG parameter)
    // property.incomingEdge = 100;                         // (ONNG parameter)

    // // WEAVES ANNG on Audio
    // property.edgeSizeLimitForCreation = 200;             // K=200 is nn and limits the number of neighbors (ANNG parameter)
    // property.edgeSizeForCreation = 200;                  // L=200 is ef_construction and limits the number of search result (ANNG parameter)
    // property.insertionRadiusCoefficient = 1.1;           // not in weaves but default (ANNG parameter)
    // property.outgoingEdge = 40;                          // (ONNG parameter)
    // property.incomingEdge = 100;                         // (ONNG parameter)

    // Deep1M
    property.edgeSizeLimitForCreation = 200;     // same as edgeSizeForCreation
    property.edgeSizeForCreation = 200;          // kc with kc > eo and kc > ei
    property.insertionRadiusCoefficient = 1.1;   // eps_c´= 0.1
    property.outgoingEdge = 30;                  // eo
    property.incomingEdge = 110;                 // ei


    std::cout << "Start creating ANNG index files" << std::endl;
    NGT::Index::create(indexPath, property);
    NGT::Index	index(indexPath);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating ANNG index files" << std::endl;

    
    std::cout << "Copy features vectores to index" << std::endl;
    for (size_t i = 0; i < count; i++) {
      auto feature_array = &features[i * dims];
      auto feature = std::vector<float>(feature_array, feature_array + dims);
      index.append(std::move(feature));
    }
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after copying features vectors to index" << std::endl;

    std::cout << "Create ANNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();
      index.enableLog();
      index.createIndex(threadNumber);
      auto time_end = std::chrono::steady_clock::now();
      auto time_per_seconds = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count();
      std::cout << "time_per_seconds " << time_per_seconds << std::endl;
    }
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating ANNG" << std::endl;


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
int refineANNGIndex(const char * anngIndexPath, const char * rnngIndexPath, const int threadNumber) {
  try {

    std::cout << "Load ANNG index files" << std::endl;
    auto index = NGT::Index(anngIndexPath);

    NGT::Property	property;
    index.getProperty(property);
    auto epsilon = property.insertionRadiusCoefficient - 1;
    auto accuracy = 0.0;  // default
    auto noOfEdges = property.edgeSizeForCreation;
    auto exploreEdgeSize = property.edgeSizeForSearch;

    // // ONNG defaults
    // // Which properties need to be set are copied from here 
    // // https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/onng_ngt.py
    // //
    // // Their values are copied from here 
    // // https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml#L378
    // //
    // // And how to use them here 
    // // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Command.cpp
    // NGT::Property	property;
    // index.getProperty(property);
    // property.indexType = NGT::Index::Property::IndexType::GraphAndTree;
    // property.threadPoolSize = 8;
    // property.batchSizeForCreation = 500;
    // property.graphType = NGT::Property::GraphType::GraphTypeANNG;
    // property.objectType	= NGT::ObjectSpace::ObjectType::Float;
    // property.distanceType	= NGT::Index::Property::DistanceType::DistanceTypeL2;
    // // property.dimension = (int) dims;
    // property.edgeSizeForCreation = 100;
    // property.outgoingEdge = 10;                 // default
    // property.incomingEdge = 120;
    // property.edgeSizeForSearch = 0;             // default
    // property.insertionRadiusCoefficient = 1.1;  // default (1.0 + epsilon)
    // property.pathAdjustmentInterval = 0;        // default
    // property.dynamicEdgeSizeBase = 30;          // default
    // property.buildTimeLimit = 4;                // hours
    // index.setProperty(property);

    // // WEAVES
    // // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
    // // NGT::Property	property;
    // // index.getProperty(property);
    // // property.edgeSizeForCreation = edgeSizeForCreation; 
    // // property.objectAlignment = NGT::Index::Property::ObjectAlignment::ObjectAlignmentTrue;
    // // property.threadPoolSize = threadNumber;
    // // property.outgoingEdge = 10;
    // // property.outgoingEdge = 120;
    // // property.dynamicEdgeSizeBase = 10;
    // // index.setProperty(property);

    // // NSG
    // // https://arxiv.org/pdf/1810.07355.pdf
    // // GraphReconstructor.h:refineANNG(...)  
    // //  -> searchContainer.setEdgeSize(exploreEdgeSize)
    // //    -> edgeSize = -1;	// dynamically prune the edges during search. -1 means following the index property inside the anngIndexPath. 0 means using all edges. any other number limits number of neighbors checked during a search
    // //  -> searchContainer.setSize(noOfSearchedEdges)
    // //    -> number of result during search
    // // ANNG edgeSizeForCreation=200 insertionRadiusCoefficient=0.1

    std::cout << "Refine ANNG to RNNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();
      //https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/onng_ngt.py
      NGT::GraphReconstructor::refineANNG(index, true, epsilon, accuracy, noOfEdges, exploreEdgeSize);
      //NGT::GraphReconstructor::refineANNG(index, true);
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
    // https://github.com/yahoojapan/NGT/blob/main/lib/NGT/Command.cpp#L709
    std::cout << "Convert ANNG to ONNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();

      // defaults
      NGT::GraphOptimizer graphOptimizer(false);
      graphOptimizer.shortcutReduction = true;              // (default=true)
      graphOptimizer.searchParameterOptimization = true;    // (default=true)
      graphOptimizer.prefetchParameterOptimization = true;  // (default=true)
      graphOptimizer.accuracyTableGeneration = false;       // does not work (default=true)
      graphOptimizer.gtEpsilon = 0.1;           // eps for searching the ground truth data (default=0.1)
      graphOptimizer.margin = 0.2;              // (default=0.2)
      graphOptimizer.minNumOfEdges =  0;        // E (default=0)
      graphOptimizer.numOfQueries = 100;        // # of ground truth objects  (default=100)
      graphOptimizer.numOfResults = 20;         // # of resultant objects     (default=20)
      graphOptimizer.numOfOutgoingEdges = 50;   // i (ONNG parameter)
      graphOptimizer.numOfIncomingEdges = 150;  // o (ONNG parameter)

      // WEAVES on SIFT1M
      // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
      // graphOptimizer.numOfOutgoingEdges = 30;        // out=30 (ONNG parameter)
      // graphOptimizer.numOfIncomingEdges = 100;       // in=100 (ONNG parameter)

      // NGT Quantization Version Paper for SIFT1M
      // https://github.com/yahoojapan/NGT/blob/main/bin/qbg/README.md
      // graphOptimizer.minNumOfEdges = 64;
      // graphOptimizer.numOfOutgoingEdges = 64;
      // graphOptimizer.numOfIncomingEdges = 120; 

      // NGT Paper for SIFT1M
      // https://arxiv.org/pdf/1810.07355.pdf 
      // graphOptimizer.numOfResults = 20; 
      // graphOptimizer.numOfOutgoingEdges = 30;
      // graphOptimizer.numOfIncomingEdges = 110;

      // NGT repo for SIFT1M
      // https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml#L378
      // https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/onng_ngt.py
      // https://github.com/yahoojapan/NGT/tree/main/bin/ngt
      // graphOptimizer.numOfOutgoingEdges = 10;
      // graphOptimizer.numOfIncomingEdges = 120;

      // our best on SIFT1M
      // graphOptimizer.numOfOutgoingEdges = 10;
      // graphOptimizer.numOfIncomingEdges = 120;
      // graphOptimizer.margin = 0.2; 
      // graphOptimizer.numOfQueries = 100;  // # of ground truth objects
      // graphOptimizer.numOfResults = 20;   // # of resultant objects

      // WEAVES on GloVe
      // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
      // graphOptimizer.numOfOutgoingEdges = 20;
      // graphOptimizer.numOfIncomingEdges = 100;

      // NGT Paper for GloVe
      // https://arxiv.org/pdf/1810.07355.pdf 
      // graphOptimizer.numOfResults = 20; 
      // graphOptimizer.numOfOutgoingEdges = 15;
      // graphOptimizer.numOfIncomingEdges = 155;

      // WEAVES on Enron
      // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
      // graphOptimizer.numOfOutgoingEdges = 20;
      // graphOptimizer.numOfIncomingEdges = 100;

      // WEAVES on Audio
      // https://github.com/Lsyhprum/WEAVESS/tree/dev/parameters
      // graphOptimizer.numOfOutgoingEdges = 40;
      // graphOptimizer.numOfIncomingEdges = 100;

      // Deep1M
      graphOptimizer.numOfResults = 20; 
      graphOptimizer.numOfOutgoingEdges = 30;
      graphOptimizer.numOfIncomingEdges = 110;


      // Parameters stored in anngIndexPath influence the optimizer
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

// SIFT1M
  // auto anngIndexPath    = R"(e:/Data/Feature/SIFT1M/NGT/anng K200 eps1.1)";
  // auto rnngIndexPath    = R"(e:/Data/Feature/SIFT1M/NGT/rnng K200 eps1.1)";
  // auto onngIndexPath    = R"(e:/Data/Feature/SIFT1M/NGT/in30 out110 noTable (anng K200 eps1.1))";
  // auto objectFile       = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";

  // GloVe
  // auto anngIndexPath    = R"(e:/Data/Feature/GloVe/NGT/anng K200 eps1.1)";
  // auto rnngIndexPath    = R"(e:/Data/Feature/GloVe/NGT/rnng K200 eps1.1)";
  // auto onngIndexPath    = R"(e:/Data/Feature/GloVe/NGT/onng in15 out155 noTable (anng K200 eps1.1))";
  // auto objectFile       = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base.fvecs)";
  
  // Enron
  // auto anngIndexPath    = R"(e:/Data/Feature/Enron/NGT/anng K200 eps1.1)";
  // auto rnngIndexPath    = R"(e:/Data/Feature/Enron/NGT/rnng K200 eps1.1)";
  // auto onngIndexPath    = R"(e:/Data/Feature/Enron/NGT/onng in20 out100 noTable (anng K200 eps1.1))";
  // auto objectFile       = R"(e:/Data/Feature/Enron/enron/enron_base.fvecs)";

  // Audio
  // auto anngIndexPath    = R"(e:/Data/Feature/Audio/NGT/anng K200 eps1.1)";
  // auto rnngIndexPath    = R"(e:/Data/Feature/Audio/NGT/rnng K200 eps1.1)";
  // auto onngIndexPath    = R"(e:/Data/Feature/Audio/NGT/onng in40 out100 noTable (anng K200 eps1.1))";
  // auto objectFile       = R"(e:/Data/Feature/Audio/audio/audio_base.fvecs)";


  // Deep1M
  auto anngIndexPath    = R"(e:/Data/Feature/Deep1M/NGT/anng K200 eps1.1)";
  auto rnngIndexPath    = R"(e:/Data/Feature/Deep1M/NGT/rnng K200 eps1.1)";
  auto onngIndexPath    = R"(e:/Data/Feature/Deep1M/NGT/in30 out110 noTable (anng K200 eps1.1))";
  auto objectFile       = R"(e:/Data/Feature/Deep1M/deep1m/deep1m_base.fvecs)";


  auto threadNumber = 1;

  // anng index construction
  int constructState = createANNGIndex(anngIndexPath, objectFile, threadNumber);
  if(constructState == 1)
    return 1;

  // anng index refinement
  // int refineState = refineANNGIndex(anngIndexPath, rnngIndexPath, threadNumber);
  // if(refineState == 1)
  //   return 1;

  int reconstructState = reconstructONNGIndex(anngIndexPath, onngIndexPath);
  if(reconstructState == 1)
    return 1;
    
  return 0;
}


