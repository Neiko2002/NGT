
#include	"NGT/Index.h"
#include	"NGT/GraphOptimizer.h"

#include <random>
#include <vector>
#include <unordered_set>
#include <filesystem>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    assert(file_size % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = file_size / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<uint32_t[]>(n * (dims + 1));
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(uint32_t));
    if (!ifstream) 
        assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) 
        memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(uint32_t));

    ifstream.close();
    return x;
}

// generate size distinct random numbers < N
template <typename RNG>
static void GenRandom(RNG &rng, uint32_t *addr, uint32_t size, uint32_t N) {
    if (N == size) {
        for (uint32_t i = 0; i < size; ++i) {
            addr[i] = i;
        }
        return;
    }
    for (uint32_t i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (uint32_t i = 1; i < size; ++i) {
        if (addr[i] <= addr[i-1]) {
            addr[i] = addr[i-1] + 1;
        }
    }
    uint32_t off = rng() % N;
    for (uint32_t i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

static uint32_t compute_reachablity_count(std::vector<std::vector<uint32_t>>& graph) {

    auto graph_size = graph.size();
    uint32_t reachable_count = 0;

    unsigned L = 100; // L_search
    unsigned seed = 1998;
    std::mt19937 rng(seed);
    std::vector<uint32_t> init_ids(L);
    GenRandom(rng, &init_ids[0], init_ids.size(), graph.size());
    
    // flood fill from this entrance position
    auto checked_ids = std::vector<bool>(graph_size);
    auto check = std::vector<uint32_t>();

    // start with the first nodes
    for (unsigned s: init_ids) {
        checked_ids[s] = true;
        check.emplace_back(s);
    }
    
    // repeat as long as we have nodes to check
	while(check.size() > 0) {	

        // neighbors which will be checked next round
        auto check_next = std::vector<uint32_t>();

        // get the neighbors to check next
        for (auto &&check_index : check) {
 
            auto& neighbor_indizies = graph[check_index];        
            auto const &neighbors = graph[check_index];

            for (int i = 0; i < neighbor_indizies.size(); i++) {
                auto neighbor_index = neighbor_indizies[i];
                
                if(checked_ids[neighbor_index] == false) {
                    checked_ids[neighbor_index] = true;
                    check_next.emplace_back(neighbor_index);
                }
            }
        }

        check = std::move(check_next);
    }

    // how many nodes have been checked
    uint32_t checked_node_count = 0;
    for (size_t i = 1; i < graph_size; i++)
        if(checked_ids[i])
            checked_node_count++;

    std::cout << "Seed Reachablity " << checked_node_count << " of " << (graph_size-1) << " vertices" << std::endl;
    return checked_node_count;
}


struct VertexReach {
  size_t vertex_id;
  uint32_t reach_count;
  std::vector<bool> reachable_ids;

  VertexReach(size_t vertex_id, uint32_t reach_count, std::vector<bool>& reachable_ids) 
        : vertex_id(vertex_id), reach_count(reach_count), reachable_ids(std::move(reachable_ids)) {}
};

static uint32_t compute_avg_reach(std::vector<std::vector<uint32_t>>& graph) {
    const auto graph_size = graph.size();
    auto time_begin = std::chrono::steady_clock::now();

    // remember those vertices which have a very high reach
    uint32_t best_vertex_reach = 0;                             
    auto vertices_reach = std::vector<VertexReach>();    
    auto index_of_vertex_reach = std::vector<uint32_t>(graph_size);    
    std::fill(index_of_vertex_reach.begin(), index_of_vertex_reach.end(), graph_size);

    // find the reach of each vertex
    uint64_t counter = 0;
    uint64_t avg_reach = 0;
    for (size_t entry_id = 1; entry_id < graph_size; entry_id++) {
        
        // flood fill from this entrance position 
        auto checked_ids = std::vector<bool>(graph_size);
        auto check = std::vector<uint32_t>();
        
        // start with the first node
        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);

        // we try to speed up the process by reaching a vertex which can reach a lot of other vertices
        uint32_t best_reach_vertex_index = 0;
        uint32_t best_reach_vertex_reach = 0;
        
        // repeat as long as we have nodes to check
		while(check.size() > 0 && best_reach_vertex_reach < graph_size-1) {	

            // neighbors which will be checked next round
            auto check_next = std::vector<uint32_t>();

            // get the neighbors to check next
            for (size_t c = 0; c < check.size() && best_reach_vertex_reach < graph_size-1; c++) {
                const auto check_index = check[c];
                const auto& neighbor_indizies = graph[check_index]; 

                if(neighbor_indizies.size() == 0)
                    std::cout << "zero out-degree for vertex " << check_index << std::endl;

                for (int n = 0; n < neighbor_indizies.size(); n++) {
                    const auto neighbor_index = neighbor_indizies[n];
                    
                    // consider only neighbors which have not been checked yet
                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next.emplace_back(neighbor_index);

                        // is the neighbor connected to a vertex which can reach a lot of other vertices
                        const auto vertex_reach_index = index_of_vertex_reach[neighbor_index];
                        if(vertex_reach_index < graph_size) {
                            const auto& neighbor_reach = vertices_reach[vertex_reach_index];

                            // found one of the best vertices or a vertex which can reach the best
                            if(neighbor_reach.reach_count > best_reach_vertex_reach) {
                                best_reach_vertex_index = vertex_reach_index;
                                best_reach_vertex_reach = neighbor_reach.reach_count;

                                // found a vertex which can reach all other vertices
                                if(neighbor_reach.reach_count == graph_size-1) 
                                   break;

                                // copy the reach of the best
                                const auto& best_vertex_checked_ids = neighbor_reach.reachable_ids;
                                for (size_t b = 0; b < best_vertex_checked_ids.size(); b++) 
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                            }
                        }
                    }
                }
            }

            check = std::move(check_next);
        }

        // found path to a vertex which can reach every other vertex
        if(best_reach_vertex_reach == graph_size-1) {
            index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            avg_reach += graph_size-1;

        } else {

            // how many nodes have been checked
            uint32_t reach_count =  0;
            for (size_t i = 1; i < graph_size; i++)
                reach_count += checked_ids[i];
            avg_reach += reach_count;
            
            // is this a new best vertex?
            if(best_vertex_reach < reach_count) {
                best_vertex_reach = reach_count;
                index_of_vertex_reach[entry_id] = (uint32_t) vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            } else if(best_reach_vertex_reach > 0) {
                index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            } else {
                index_of_vertex_reach[entry_id] = (uint32_t) vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            }
        }

        counter++;
        if(counter % 10000 == 0) {
            const auto time_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - time_begin).count();
            std::cout << string_format("Avg reach is %.2f after checking %7d of %7d vertices after %4ds\n", ((float)avg_reach)/counter, counter, graph_size-1, time_sec);
        }
    }  

    const auto time_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - time_begin).count();
    std::cout << string_format("Avg reach is %.2f after checking %7d of %7d vertices after %4ds\n", ((float)avg_reach)/counter, counter, graph_size-1, time_sec);
    return (uint32_t)(avg_reach/counter);
}

static std::vector<std::vector<uint32_t>> getCompactGraph(NGT::GraphIndex& graphIndex) {
    const size_t size =  graphIndex.repository.size();
    auto compact_graph = std::vector<std::vector<uint32_t>>(size);

    for (size_t id = 1; id < size; id++) {
      const auto vertex = graphIndex.getNode(id);

        auto& neighbor_ids = compact_graph[id];
        for (size_t i = 0; i < vertex->size(); i++) {
            auto neighbor_id = (*vertex)[i].id;
            neighbor_ids.push_back(neighbor_id);
        }
    }
    return compact_graph;
}

static std::vector<uint32_t> computePerfectNeighborsOf(NGT::GraphIndex& graphIndex, size_t target_id, size_t k) {
    const size_t size = graphIndex.repository.size();
    NGT::ObjectSpace::Comparator &comparator = graphIndex.objectSpace->getComparator();
    NGT::ObjectRepository &objectRepository = graphIndex.getObjectRepository();

    NGT::ObjectDistance objDist;
    NGT::NeighborhoodGraph::ResultSet distances;

    NGT::Object fv = *objectRepository.get(target_id);
    for (size_t id = 1; id < size; id++) {
        auto distance = comparator(fv, *objectRepository.get(id));
        objDist.set(id, distance);
        distances.push(objDist);
    }

    auto result = std::vector<uint32_t>(k);
    for (size_t i = 0; i < k; i++) {
        result[i] = distances.top().id;
        distances.pop();
    }
    return std::move(result);
}

static void compute_stats(const char* graph_file, const uint32_t feature_dims, const char* top_list_file) {
    std::cout << "Compute graph stats of " << graph_file << std::endl;

    auto index = NGT::Index(graph_file);
    auto graphIndex = (NGT::GraphIndex&) index.getIndex();
    auto graph = getCompactGraph(graphIndex);
    auto graph_size = graph.size();
    std::cout << "graph_size=" << (graph_size-1) << std::endl;

    // compute the graph quality
    float perfect_neighbor_ratio = 0;
    float avg_edge_count = 0;
    uint64_t oversize_neighborhood = 0;
    {
        size_t top_list_dims;
        size_t top_list_count;
        const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
        std::cout << "Load TopList from file" << top_list_file << " with " << top_list_count << " elements and k=" << top_list_dims << std::endl;

        uint64_t perfect_neighbor_count = 0;
        uint64_t total_neighbor_count = 0;
        for (uint32_t n = 1; n < graph_size; n++) {
            auto& neighbor_indizies = graph[n];
            const auto edges_per_node = neighbor_indizies.size();

            // get top list of this node
            auto top_list = std::vector<uint32_t>(all_top_list.get() + (n-1) * top_list_dims, all_top_list.get() + n * top_list_dims);
            if(top_list_dims < edges_per_node) {
                top_list = std::move(computePerfectNeighborsOf(graphIndex, n, edges_per_node));
                oversize_neighborhood++;
            }
            total_neighbor_count += edges_per_node;

            // check if every neighbor is from the perfect neighborhood
            for (uint32_t e = 0; e < edges_per_node; e++) {
                auto neighbor_index = neighbor_indizies[e] - 1; // ids start at 1 not 0 in the graph

                // find in the neighbor ini the first few elements of the top list
                for (uint32_t i = 0; i < edges_per_node; i++) {
                    if(neighbor_index == top_list[i]) {
                        perfect_neighbor_count++;
                        break;
                    }
                }
            }
        }

        perfect_neighbor_ratio = ((float) perfect_neighbor_count) / total_neighbor_count;
        avg_edge_count = ((float) total_neighbor_count) / graph_size;
    }

    // compute the min and max out degree
    uint32_t min_out = std::numeric_limits<uint32_t>::max();
    uint32_t max_out = 0;
    for (uint32_t n = 1; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }

      // compute the in_degree per vertex
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 1; n < graph_size; n++) {
        auto& neighbor_indizies = graph[n];
        auto edges_per_node = neighbor_indizies.size();

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];
            in_degree_count[neighbor_index]++;
        }
    }

    // compute the min and max in degree
    uint32_t min_in = std::numeric_limits<uint32_t>::max();
    uint32_t max_in = 0;
    uint32_t source_nodes = 0;
    for (uint32_t n = 1; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;
        if(in_degree == 0) 
            source_nodes++;
    }
    std::printf("GQ %.4f, avg degree %.1f, min_out %d, max_out %d, min_in %d, max_in %d, source vertices %d, vertex count %zd, oversize neighborhood %zd\n", perfect_neighbor_ratio, avg_edge_count, min_out, max_out, min_in, max_in, source_nodes, graph_size-1, oversize_neighborhood);


    auto reachability_count = compute_reachablity_count(graph);
    auto avg_reach = compute_avg_reach(graph);
    std::printf("search reachability count %d, exploration avg reach %d\n", reachability_count, avg_reach);
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

    // ----------------------------------------- SIFT1M ------------------------------------------------
    // const auto dims = 128;
    // const auto top_list_file  = R"(e:/Data/Feature/SIFT1M/SIFT1M/sift_base_top1000.ivecs)";
    // const auto indexPath      = R"(e:/Data/Feature/SIFT1M/NGT/in30 out110 noTable (anng200 eps1.1))";

    // const auto indexPath     = R"(e:/Data/Feature/SIFT1M/NGT/anng100_onng10_120_default)";  // GQ 0.973031, PNR 0.351096, avg degree 49.3349, min_out 4,  max_out 626, min_in 18, max_in 143, zero in nodes 0
    // const auto indexPath     = R"(e:/Data/Feature/SIFT1M/NGT/anng500-onng50_150_default)";        // GQ 0.98296,  PNR 0.405321, avg degree 63.0182, min_out 15, max_out 531, min_in 23, max_in 318, zero in nodes 0

    // ----------------------------------------- Glove ------------------------------------------------
    // const auto dims = 100;
    // const auto top_list_file = R"(e:/Data/Feature/GloVe/glove-100/glove-100_base_top1000.ivecs)";
    // const auto indexPath     = R"(e:/Data/Feature/GloVe/NGT/onng in15 out155 noTable (anng K200 eps1.1))";

    // ----------------------------------------- Enron ------------------------------------------------
    // const auto dims = 1368;
    // const auto top_list_file = R"(e:/Data/Feature/Enron/enron/enron_base_top1000.ivecs)";
    // const auto indexPath     = R"(e:/Data/Feature/Enron/NGT/onng in20 out100 noTable (anng K200 eps1.1))";

    // ----------------------------------------- Audio ------------------------------------------------
    const auto dims = 192;
    const auto top_list_file = R"(e:/Data/Feature/Audio/audio/audio_base_top1000.ivecs)";
    const auto indexPath     = R"(e:/Data/Feature/Audio/NGT/onng in40 out100 noTable (anng K200 eps1.1))";

    compute_stats(indexPath, dims, top_list_file);

    std::cout << "Finished" << std::endl;
}