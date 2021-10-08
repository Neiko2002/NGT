
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
int createANNGIndex(const char * indexPath, const char * featurePath, const int edgeSizeForCreation) {
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
      index.createIndex(1);
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
int refineANNGIndex(const char * anngIndexPath, const char * rnngIndexPath, const int edgeSizeForCreation) {
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
    property.threadPoolSize = 1;
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
 

  auto anngIndexPath    = R"(c:/Data/Feature/SIFT1M/NGT/anng)";
  auto rnngIndexPath    = R"(c:/Data/Feature/SIFT1M/NGT/rnng1)";
  auto objectFile       = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  auto queryFile        = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_query.fvecs)";
  auto groundtruthFile	= R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_groundtruth.ivecs)";
  
  auto edgeSizeForCreation = 100;

  // anng index construction
  /*int state = createANNGIndex(anngIndexPath, objectFile, edgeSizeForCreation);
  if(state == 1)
    return 1;*/

  // anng index refinement
  int refineState = refineANNGIndex(anngIndexPath, rnngIndexPath, edgeSizeForCreation);
  if(refineState == 1)
    return 1;

/*
  try {

    // ngt reconstruct-graph -m S -o outdegree -i indegree anng-index onng-index
    // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Command.cpp#L700
    std::cout << "Convert ANNG to ONNG" << std::endl;
    {
      auto time_begin = std::chrono::steady_clock::now();
*/
      /*
      NGT::GraphOptimizer graphOptimizer(false);
      graphOptimizer.shortcutReduction = true;
      graphOptimizer.searchParameterOptimization = true;
      graphOptimizer.prefetchParameterOptimization = true;
      graphOptimizer.accuracyTableGeneration = true;
      graphOptimizer.margin = 0.2;
      graphOptimizer.gtEpsilon = 0.1;
      graphOptimizer.minNumOfEdges =  0;


      size_t nOfQueries = 100;		// # of query objects
      size_t nOfResults =  20;		// # of resultant objects

      int numOfOutgoingEdges	= args.getl("o", -1);
      int numOfIncomingEdges	= args.getl("i", -1);

      graphOptimizer.set(numOfOutgoingEdges, numOfIncomingEdges, nOfQueries, nOfResults);
      graphOptimizer.execute(inIndexPath, outIndexPath);
      */

/*
      NGT::GraphReconstructor::reconstructGraph(index, true);
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
*/


/*
  // nearest neighbor search
  try {
    NGT::Index		index(indexPath);
    NGT::Property	property;
    index.getProperty(property);
    
    std::ifstream		is(queryFile);
    std::string		line;
    while (getline(is, line)) {
      std::vector<uint8_t>	query;
      {
        std::stringstream	linestream(line);
        while (!linestream.eof()) {
          int value;
          linestream >> value;
          query.push_back(value);
        }
        query.resize(property.dimension);
        cout << "Query : ";
        for (size_t i = 0; i < 5; i++) {
          std::cout << static_cast<int>(query[i]) << " ";
        }
        std::cout << "...";
      }

      NGT::SearchQuery		sc(query);
      NGT::ObjectDistances	objects;
      sc.setResults(&objects);
      sc.setSize(10);
      sc.setEpsilon(0.1f);

      index.search(sc);
      std::cout << endl << "Rank\tID\tDistance" << std::showbase << endl;
      for (size_t i = 0; i < objects.size(); i++) {
        std::cout << i + 1 << "\t" << objects[i].id << "\t" << objects[i].distance << "\t: ";
        NGT::ObjectSpace &objectSpace = index.getObjectSpace();
        uint8_t *object = static_cast<uint8_t*>(objectSpace.getObject(objects[i].id));
        for (size_t idx = 0; idx < 5; idx++) {
          std::cout << static_cast<int>(object[idx]) << " ";
        }
        std::cout << "..." << endl;
      }
      std::cout << endl;
    }
  } catch (NGT::Exception &err) {
    std::cerr << "Error " << err.what() << endl;
    return 1;
  } catch (...) {
    std::cerr << "Error" << endl;
    return 1;
  }
*/
  return 0;
}


