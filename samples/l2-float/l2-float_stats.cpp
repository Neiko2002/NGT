
#include	"NGT/Index.h"
#include	"NGT/GraphOptimizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <assert.h>


#include <vector>
#include <unordered_set>
static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        std::cerr << "error when accessing top list file" << fname << " size is: " << file_size << " message: " << ec.message() << std::endl;
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::cerr << "could not open " << fname << std::endl;
        perror("");
        abort();
    }

    uint32_t dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert((file_size - 4) % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = (file_size - 4) / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<uint32_t[]>(n * (dims + 1));
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(uint32_t));
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(uint32_t));

    ifstream.close();
    return x;
}

static void compute_stats(const char* graph_file, const uint32_t feature_dims, const char* top_list_file) {
    std::cout << "Compute graph stats of " << graph_file << std::endl;

    size_t top_list_dims;
    size_t top_list_count;
    const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
    std::cout << "Load TopList from file" << top_list_file << " with " << top_list_count << " elements and k=" << top_list_dims << std::endl;

    auto index = NGT::Index(graph_file);
    auto graph = (NGT::GraphIndex&) index.getIndex();
    auto graph_size = graph.repository.size();
     std::cout << "graph_size" << graph_size << std::endl;

    /*for (size_t id = 1; id < graph.repository.size(); id++) {
      auto node = graph.getNode(id);

      for (size_t i = 0; i < node->size(); i++) {
        auto neighbor_id = (*node)[i].id;
      }
    }*/
    
    // compute the graph quality
    uint64_t perfect_neighbor_count = 0;
    uint64_t total_neighbor_count = 0;
    for (uint32_t n = 1; n < graph_size; n++) {
        auto node = graph.getNode(n);
        auto edges_per_node = node->size();

        // get top list of this node
        auto top_list = all_top_list.get() + (n-1) * top_list_dims;
        if(top_list_dims < edges_per_node) {
            std::cerr << "TopList for " << (n-1) << " is not long enough has " << edges_per_node << " elements has " << top_list_dims << std::endl;
            edges_per_node = (uint16_t) top_list_dims;
        }
        total_neighbor_count += edges_per_node;

        // check if every neighbor is from the perfect neighborhood
        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = (*node)[e].id - 1;

            // find in the neighbor ini the first few elements of the top list
            for (uint32_t i = 0; i < edges_per_node; i++) {
                if(neighbor_index == top_list[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
    }
    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / total_neighbor_count;
    auto avg_edge_count = (float) total_neighbor_count / graph_size;

    // compute the min, and max out degree
    uint16_t min_out =  std::numeric_limits<uint16_t>::max();
    uint16_t max_out = 0;
    for (uint32_t n = 1; n < graph_size; n++) {
        auto node = graph.getNode(n);
        auto edges_per_node = node->size();

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }

    // compute the min, and max in degree
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 1; n < graph_size; n++) {
        auto node = graph.getNode(n);
        auto edges_per_node = node->size();

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = (*node)[e].id;
            in_degree_count[neighbor_index]++;
        }
    }

    uint32_t min_in = std::numeric_limits<uint32_t>::max();
    uint32_t max_in = 0;
    uint32_t zero_in_count = 0;
    for (uint32_t n = 1; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;

        if(in_degree == 0) {
            zero_in_count++;
            std::cout << "Node " << n << " has zero incoming connections" << std::endl;
        }
    }

    std::cout << "GQ " << perfect_neighbor_ratio << ", avg degree " << avg_edge_count << ", min_out " << min_out << ", max_out " << max_out << ", min_in " << min_in << ", max_in " << max_in << ", zero in nodes " << zero_in_count << "\n" << std::endl;
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

    auto indexPath     = R"(c:/Data/Feature/SIFT1M/NGT/anng500-onng50_150_default)";  // GQ 0.405493, avg degree 62.9719, min_out 15, max_out 531, min_in 23, max_in 318, zero in nodes 0
    auto top_list_file = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base_top200_p0.998.ivecs)";
    compute_stats(indexPath, 128, top_list_file);

    std::cout << "Finished" << std::endl;
}