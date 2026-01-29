#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssg::benchmark {

inline std::unique_ptr<uint32_t[]> ivecs_read(const char* filename, size_t& d_out, size_t& n_out) {
    d_out = 0;
    n_out = 0;

    std::error_code ec{};
    const auto file_size = std::filesystem::file_size(filename, ec);
    if (ec != std::error_code{}) {
        return nullptr;
    }

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        return nullptr;
    }

    uint32_t dims = 0;
    in.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    if (!in) {
        return nullptr;
    }

    if (dims == 0 || dims > 1'000'000) {
        return nullptr;
    }

    const size_t row_bytes = (static_cast<size_t>(dims) + 1) * sizeof(uint32_t);
    if (row_bytes == 0 || (file_size % row_bytes) != 0) {
        return nullptr;
    }

    const size_t n = static_cast<size_t>(file_size / row_bytes);
    d_out = dims;
    n_out = n;

    auto file_bytes = std::make_unique<std::byte[]>(static_cast<size_t>(file_size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(file_bytes.get()), static_cast<std::streamsize>(file_size));
    if (!in) {
        return nullptr;
    }

    auto data = std::make_unique<uint32_t[]>(n_out * d_out);
    for (size_t i = 0; i < n_out; ++i) {
        const std::byte* row = file_bytes.get() + i * row_bytes;
        uint32_t dim_check = 0;
        std::memcpy(&dim_check, row, sizeof(uint32_t));
        if (dim_check != dims) {
            d_out = 0;
            n_out = 0;
            return nullptr;
        }

        std::memcpy(data.get() + i * d_out, row + sizeof(uint32_t), d_out * sizeof(uint32_t));
    }

    return data;
}

// Load fvecs file into raw float pointer
// Returns: pointer to data (caller owns), sets num and dim
inline float* load_fvecs(const char* filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(std::string("Could not open file: ") + filename);
    }

    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (!in || dim <= 0) {
        throw std::runtime_error("Invalid fvecs header");
    }

    in.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    const size_t vec_size = sizeof(int) + static_cast<size_t>(dim) * sizeof(float);
    num = static_cast<unsigned>(file_size / vec_size);

    float* data = new float[static_cast<size_t>(num) * dim];

    for (size_t i = 0; i < num; ++i) {
        int d = 0;
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (d != static_cast<int>(dim)) {
            delete[] data;
            throw std::runtime_error("Dimension mismatch in fvecs file");
        }
        in.read(reinterpret_cast<char*>(data + i * dim), static_cast<std::streamsize>(dim * sizeof(float)));
    }

    return data;
}

inline bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

inline bool file_exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

inline bool ensure_directory(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) return true;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
}

}  // namespace ssg::benchmark
