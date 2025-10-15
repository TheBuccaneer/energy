// spmv_sweep.cu - cuSPARSE SpMV Benchmark with Size Sweep and NVML Energy
// VERSION: 2.0.0-FIXED (2025-01-15) - CSV Bug Fixed + 8 Patterns
// Compile: nvcc -O3 -std=c++17 -lcusparse -lnvidia-ml -o spmv_sweep spmv_sweep.cu

#include <cuda_runtime.h>
#include <cusparse.h>
#include <nvml.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <random>
#include <numeric>

// ============================================================================
// Data Type Configuration
// ============================================================================

using real = float;
constexpr const char* DTYPE_STR = "fp32";

// ============================================================================
// Configuration
// ============================================================================

// Matrix sizes (total rows) - will generate N×N sparse matrices where N = sqrt(size)
static constexpr size_t MATRIX_SIZES[] = {
    1024,      // 32×32 grid = 1K rows
    4096,      // 64×64 grid = 4K rows
    16384,     // 128×128 grid = 16K rows
    65536,     // 256×256 grid = 64K rows
    131072,    // 362×362 grid ≈ 128K rows
    262144     // 512×512 grid = 256K rows
};

// Sparsity patterns
enum SparsityPattern {
    STENCIL_5PT,        // ~5 non-zeros per row (2D structured)
    IRREGULAR_20,       // ~20 non-zeros per row (random)
    STENCIL_7PT_3D,     // ~7 non-zeros per row (3D structured)
    STENCIL_27PT_3D,    // ~27 non-zeros per row (3D structured)
    BAND_TRIDIAG,       // Tridiagonal band matrix
    BLOCK_DIAGONAL,     // Dense blocks along diagonal
    RMAT_POWERLAW       // R-MAT/Graph500-style power-law
};

// Pattern specification for runs
struct PatternSpec {
    SparsityPattern pat;
    bool permuted;
    int band_width;
    int block_size;
    int scale;
    int edge_factor;
    const char* name;
};

// Hardcoded run configurations
static constexpr PatternSpec RUNS[] = {
    { STENCIL_5PT,      false, 1, 0, 0,  0,  "STENCIL_5PT_2D" },
    { IRREGULAR_20,     false, 0, 0, 0,  0,  "IRREGULAR_20" },
    { STENCIL_7PT_3D,   false, 1, 0, 0,  0,  "STENCIL_7PT_3D" },
    { STENCIL_27PT_3D,  false, 1, 0, 0,  0,  "STENCIL_27PT_3D" },
    { BAND_TRIDIAG,     false, 1, 0, 0,  0,  "BAND_TRIDIAG" },
    { BLOCK_DIAGONAL,   false, 0, 8, 0,  0,  "BLOCK_DIAGONAL" },
    { RMAT_POWERLAW,    false, 0, 0, 20, 16, "RMAT_POWERLAW" },
    { RMAT_POWERLAW,    true,  0, 0, 20, 16, "RMAT_POWERLAW_PERM" }
};

static constexpr int REPEATS = 50;
static constexpr double TARGET_S = 1.1;
static constexpr double SAFETY_FACTOR = 1.02;
static constexpr double TARGET_LOW = 0.95;  // Lower bound of target envelope
static constexpr size_t MIN_PASSES_SMALL = 100;  // For very small matrices
static constexpr double OOM_SAFETY = 0.60;  // Use 60% of free VRAM

const char* CSV_PATH = "data/raw/spmv_sweep.csv";

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call) do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
    cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
} \
} while(0)

#define CUSPARSE_CHECK(call) do { \
cusparseStatus_t err = call; \
if (err != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSPARSE Error at %s:%d: %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
} \
} while(0)

#define NVML_CHECK(call) do { \
nvmlReturn_t err = call; \
if (err != NVML_SUCCESS) { \
    fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__, \
    nvmlErrorString(err)); \
} \
} while(0)

// ============================================================================
// NVML Helper
// ============================================================================

struct NVMLContext {
    nvmlDevice_t device;
    bool initialized = false;
    bool energy_supported = false;

    bool init() {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "NVML Init failed: " << nvmlErrorString(result) << "\n";
            return false;
        }

        result = nvmlDeviceGetHandleByIndex(0, &device);
        if (result != NVML_SUCCESS) {
            std::cerr << "Failed to get NVML device handle\n";
            nvmlShutdown();
            return false;
        }

        unsigned long long test_energy;
        result = nvmlDeviceGetTotalEnergyConsumption(device, &test_energy);
        energy_supported = (result == NVML_SUCCESS);

        initialized = true;
        return true;
    }

    unsigned long long getTotalEnergyMilliJ() {
        if (!initialized || !energy_supported) return 0;
        unsigned long long energy_mj = 0;
        nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
        if (result != NVML_SUCCESS) return 0;
        return energy_mj;  // Returns millijoules since driver reload
    }

    void shutdown() {
        if (initialized) {
            nvmlShutdown();
            initialized = false;
        }
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

std::string getTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

std::string getHostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
}

void ensureDirectoryExists(const char* filepath) {
    namespace fs = std::filesystem;
    fs::path file_path(filepath);
    if (file_path.has_parent_path()) {
        fs::create_directories(file_path.parent_path());
    }
}

bool fileExists(const char* filepath) {
    struct stat buffer;
    return (stat(filepath, &buffer) == 0);
}

std::string readableBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

// ============================================================================
// Device Info
// ============================================================================

struct DeviceInfo {
    std::string name;
    size_t total_global_mem;
    int cc_major;
    int cc_minor;
    int driver_version;
    int sm_count;
};

DeviceInfo getDeviceInfo() {
    DeviceInfo info;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    info.name = std::string(prop.name);
    info.total_global_mem = prop.totalGlobalMem;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;

    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));

    return info;
}

// ============================================================================
// GPU Runtime Info
// ============================================================================

struct GPURuntimeInfo {
    unsigned int temp_c = 0;
    unsigned int clocks_sm_mhz = 0;
    unsigned int clocks_mem_mhz = 0;
    unsigned long long throttle_reasons = 0;
    unsigned int pcie_gen = 0;
    unsigned int pcie_width = 0;
    unsigned int pcie_rx_kbs = 0;
    unsigned int pcie_tx_kbs = 0;
};

GPURuntimeInfo getGPURuntimeInfo(nvmlDevice_t device, bool nvml_ok) {
    GPURuntimeInfo info;
    if (!nvml_ok) return info;

    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temp_c);
    nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &info.clocks_sm_mhz);
    nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &info.clocks_mem_mhz);
    nvmlDeviceGetCurrentClocksThrottleReasons(device, &info.throttle_reasons);
    nvmlDeviceGetCurrPcieLinkGeneration(device, &info.pcie_gen);
    nvmlDeviceGetCurrPcieLinkWidth(device, &info.pcie_width);
    nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &info.pcie_rx_kbs);
    nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &info.pcie_tx_kbs);

    return info;
}

// ============================================================================
// CSR Matrix Structure
// ============================================================================

struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<real> values;

    double avg_nnz_per_row = 0.0;
    double std_nnz_per_row = 0.0;

    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    real* d_values = nullptr;

    void computeStatistics() {
        if (rows == 0) return;

        std::vector<int> nnz_per_row(rows);
        for (int i = 0; i < rows; i++) {
            nnz_per_row[i] = row_ptr[i+1] - row_ptr[i];
        }

        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += nnz_per_row[i];
        }
        avg_nnz_per_row = sum / rows;

        double var_sum = 0.0;
        for (int i = 0; i < rows; i++) {
            double diff = nnz_per_row[i] - avg_nnz_per_row;
            var_sum += diff * diff;
        }
        std_nnz_per_row = std::sqrt(var_sum / rows);
    }

    void allocateDevice() {
        CUDA_CHECK(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(real)));
    }

    void copyToDevice() {
        CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(real), cudaMemcpyHostToDevice));
    }

    void freeDevice() {
        if (d_row_ptr) { cudaFree(d_row_ptr); d_row_ptr = nullptr; }
        if (d_col_idx) { cudaFree(d_col_idx); d_col_idx = nullptr; }
        if (d_values) { cudaFree(d_values); d_values = nullptr; }
    }

    size_t deviceMemoryBytes() const {
        return (rows + 1) * sizeof(int) + nnz * sizeof(int) + nnz * sizeof(real);
    }
};

// ============================================================================
// Matrix Generation
// ============================================================================

CSRMatrix generateStencil5pt(int N) {
    CSRMatrix mat;
    mat.rows = N * N;
    mat.cols = N * N;
    mat.row_ptr.resize(mat.rows + 1);

    std::vector<std::vector<int>> temp_cols(mat.rows);
    std::vector<std::vector<real>> temp_vals(mat.rows);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int row = i * N + j;

            // Center
            temp_cols[row].push_back(row);
            temp_vals[row].push_back(4.0f);

            // Left
            if (j > 0) {
                temp_cols[row].push_back(row - 1);
                temp_vals[row].push_back(-1.0f);
            }
            // Right
            if (j < N - 1) {
                temp_cols[row].push_back(row + 1);
                temp_vals[row].push_back(-1.0f);
            }
            // Up
            if (i > 0) {
                temp_cols[row].push_back(row - N);
                temp_vals[row].push_back(-1.0f);
            }
            // Down
            if (i < N - 1) {
                temp_cols[row].push_back(row + N);
                temp_vals[row].push_back(-1.0f);
            }
        }
    }

    int offset = 0;
    mat.row_ptr[0] = 0;
    for (int row = 0; row < mat.rows; row++) {
        for (size_t k = 0; k < temp_cols[row].size(); k++) {
            mat.col_idx.push_back(temp_cols[row][k]);
            mat.values.push_back(temp_vals[row][k]);
        }
        offset += temp_cols[row].size();
        mat.row_ptr[row + 1] = offset;
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

CSRMatrix generateIrregular20(int N) {
    CSRMatrix mat;
    mat.rows = N * N;
    mat.cols = N * N;
    mat.row_ptr.resize(mat.rows + 1);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nnz_dist(15, 25);
    std::uniform_int_distribution<int> col_dist(0, mat.cols - 1);
    std::uniform_real_distribution<real> val_dist(0.1f, 1.0f);

    int offset = 0;
    mat.row_ptr[0] = 0;

    for (int row = 0; row < mat.rows; row++) {
        int nnz_row = std::min(nnz_dist(rng), mat.cols);
        std::vector<int> cols;

        // Always include diagonal
        cols.push_back(row);

        // Add random columns
        while (static_cast<int>(cols.size()) < nnz_row) {
            int col = col_dist(rng);
            if (std::find(cols.begin(), cols.end(), col) == cols.end()) {
                cols.push_back(col);
            }
        }

        std::sort(cols.begin(), cols.end());

        for (int col : cols) {
            mat.col_idx.push_back(col);
            mat.values.push_back(val_dist(rng));
        }

        offset += cols.size();
        mat.row_ptr[row + 1] = offset;
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

CSRMatrix generateStencil7pt3D(int total_rows) {
    // Compute cube dimensions
    int n = static_cast<int>(std::cbrt(total_rows) + 0.5);
    int actual_rows = n * n * n;

    CSRMatrix mat;
    mat.rows = actual_rows;
    mat.cols = actual_rows;
    mat.row_ptr.resize(mat.rows + 1);

    std::vector<std::vector<int>> temp_cols(mat.rows);
    std::vector<std::vector<real>> temp_vals(mat.rows);

    auto idx = [n](int i, int j, int k) { return i * n * n + j * n + k; };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                int row = idx(i, j, k);

                // Center
                temp_cols[row].push_back(row);
                temp_vals[row].push_back(6.0f);

                // -x
                if (i > 0) {
                    temp_cols[row].push_back(idx(i-1, j, k));
                    temp_vals[row].push_back(-1.0f);
                }
                // +x
                if (i < n-1) {
                    temp_cols[row].push_back(idx(i+1, j, k));
                    temp_vals[row].push_back(-1.0f);
                }
                // -y
                if (j > 0) {
                    temp_cols[row].push_back(idx(i, j-1, k));
                    temp_vals[row].push_back(-1.0f);
                }
                // +y
                if (j < n-1) {
                    temp_cols[row].push_back(idx(i, j+1, k));
                    temp_vals[row].push_back(-1.0f);
                }
                // -z
                if (k > 0) {
                    temp_cols[row].push_back(idx(i, j, k-1));
                    temp_vals[row].push_back(-1.0f);
                }
                // +z
                if (k < n-1) {
                    temp_cols[row].push_back(idx(i, j, k+1));
                    temp_vals[row].push_back(-1.0f);
                }
            }
        }
    }

    int offset = 0;
    mat.row_ptr[0] = 0;
    for (int row = 0; row < mat.rows; row++) {
        for (size_t p = 0; p < temp_cols[row].size(); p++) {
            mat.col_idx.push_back(temp_cols[row][p]);
            mat.values.push_back(temp_vals[row][p]);
        }
        offset += temp_cols[row].size();
        mat.row_ptr[row + 1] = offset;
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

CSRMatrix generateStencil27pt3D(int total_rows) {
    int n = static_cast<int>(std::cbrt(total_rows) + 0.5);
    int actual_rows = n * n * n;

    CSRMatrix mat;
    mat.rows = actual_rows;
    mat.cols = actual_rows;
    mat.row_ptr.resize(mat.rows + 1);

    std::vector<std::vector<int>> temp_cols(mat.rows);
    std::vector<std::vector<real>> temp_vals(mat.rows);

    auto idx = [n](int i, int j, int k) { return i * n * n + j * n + k; };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                int row = idx(i, j, k);

                // 27-point stencil: all neighbors in 3x3x3 cube
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        for (int dk = -1; dk <= 1; dk++) {
                            int ni = i + di;
                            int nj = j + dj;
                            int nk = k + dk;

                            if (ni >= 0 && ni < n && nj >= 0 && nj < n && nk >= 0 && nk < n) {
                                int col = idx(ni, nj, nk);
                                temp_cols[row].push_back(col);
                                temp_vals[row].push_back((di == 0 && dj == 0 && dk == 0) ? 26.0f : -1.0f);
                            }
                        }
                    }
                }
            }
        }
    }

    int offset = 0;
    mat.row_ptr[0] = 0;
    for (int row = 0; row < mat.rows; row++) {
        for (size_t p = 0; p < temp_cols[row].size(); p++) {
            mat.col_idx.push_back(temp_cols[row][p]);
            mat.values.push_back(temp_vals[row][p]);
        }
        offset += temp_cols[row].size();
        mat.row_ptr[row + 1] = offset;
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

CSRMatrix generateBandTridiag(int total_rows, int band_width) {
    CSRMatrix mat;
    mat.rows = total_rows;
    mat.cols = total_rows;
    mat.row_ptr.resize(mat.rows + 1);

    int offset = 0;
    mat.row_ptr[0] = 0;

    for (int row = 0; row < mat.rows; row++) {
        for (int bw = -band_width; bw <= band_width; bw++) {
            int col = row + bw;
            if (col >= 0 && col < mat.cols) {
                mat.col_idx.push_back(col);
                mat.values.push_back((bw == 0) ? 2.0f : -1.0f);
                offset++;
            }
        }
        mat.row_ptr[row + 1] = offset;
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

CSRMatrix generateBlockDiagonal(int total_rows, int block_size) {
    CSRMatrix mat;
    mat.rows = total_rows;
    mat.cols = total_rows;
    mat.row_ptr.resize(mat.rows + 1);

    std::mt19937 rng(42);
    std::uniform_real_distribution<real> val_dist(0.1f, 1.0f);

    int num_blocks = (total_rows + block_size - 1) / block_size;
    int offset = 0;
    mat.row_ptr[0] = 0;

    for (int block = 0; block < num_blocks; block++) {
        int block_start = block * block_size;
        int block_end = std::min(block_start + block_size, total_rows);

        for (int row = block_start; row < block_end; row++) {
            for (int col = block_start; col < block_end; col++) {
                mat.col_idx.push_back(col);
                mat.values.push_back(val_dist(rng));
                offset++;
            }
            mat.row_ptr[row + 1] = offset;
        }
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

CSRMatrix generateRMAT(int total_rows, int scale, int edge_factor) {
    CSRMatrix mat;

    // Use scale if provided, otherwise infer from total_rows
    int n_bits = (scale > 0) ? scale : static_cast<int>(std::ceil(std::log2(total_rows)));
    int n = 1 << n_bits;  // n = 2^scale

    mat.rows = n;
    mat.cols = n;

    // R-MAT parameters
    const double a = 0.57, b = 0.19, c = 0.19; // d = 1-a-b-c

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<real> val_dist(0.1f, 1.0f);

    // Generate edges using R-MAT: m = n * edge_factor
    int num_edges = n * edge_factor;
    std::vector<std::pair<int, int>> edges;

    for (int e = 0; e < num_edges; e++) {
        int u = 0, v = 0;

        for (int bit = 0; bit < n_bits; bit++) {
            double p = prob_dist(rng);
            int u_bit = 0, v_bit = 0;

            if (p < a) {
                u_bit = 0; v_bit = 0;
            } else if (p < a + b) {
                u_bit = 0; v_bit = 1;
            } else if (p < a + b + c) {
                u_bit = 1; v_bit = 0;
            } else {
                u_bit = 1; v_bit = 1;
            }

            u = (u << 1) | u_bit;
            v = (v << 1) | v_bit;
        }

        if (u != v) {
            edges.push_back({u, v});
            edges.push_back({v, u}); // Symmetrize
        }
    }

    // Sort and remove duplicates
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // Build CSR
    std::vector<std::vector<int>> temp_cols(mat.rows);
    std::vector<std::vector<real>> temp_vals(mat.rows);

    for (const auto& edge : edges) {
        temp_cols[edge.first].push_back(edge.second);
        temp_vals[edge.first].push_back(val_dist(rng));
    }

    mat.row_ptr.resize(mat.rows + 1);
    int offset = 0;
    mat.row_ptr[0] = 0;

    for (int row = 0; row < mat.rows; row++) {
        std::sort(temp_cols[row].begin(), temp_cols[row].end());
        for (size_t p = 0; p < temp_cols[row].size(); p++) {
            mat.col_idx.push_back(temp_cols[row][p]);
            mat.values.push_back(temp_vals[row][p]);
        }
        offset += temp_cols[row].size();
        mat.row_ptr[row + 1] = offset;
    }

    mat.nnz = mat.col_idx.size();
    mat.computeStatistics();
    return mat;
}

// Permute CSR matrix (P*A*P^T)
void permuteCSR(CSRMatrix& mat) {
    std::mt19937 rng(123);
    std::vector<int> perm(mat.rows);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<int> inv_perm(mat.rows);
    for (int i = 0; i < mat.rows; i++) {
        inv_perm[perm[i]] = i;
    }

    // Build new CSR
    CSRMatrix new_mat;
    new_mat.rows = mat.rows;
    new_mat.cols = mat.cols;
    new_mat.row_ptr.resize(mat.rows + 1);

    std::vector<std::vector<int>> temp_cols(mat.rows);
    std::vector<std::vector<real>> temp_vals(mat.rows);

    for (int old_row = 0; old_row < mat.rows; old_row++) {
        int new_row = inv_perm[old_row];
        for (int p = mat.row_ptr[old_row]; p < mat.row_ptr[old_row + 1]; p++) {
            int old_col = mat.col_idx[p];
            int new_col = inv_perm[old_col];
            temp_cols[new_row].push_back(new_col);
            temp_vals[new_row].push_back(mat.values[p]);
        }
    }

    int offset = 0;
    new_mat.row_ptr[0] = 0;
    for (int row = 0; row < mat.rows; row++) {
        auto& cols = temp_cols[row];
        auto& vals = temp_vals[row];

        // Sort by column
        std::vector<int> idx(cols.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&cols](int a, int b) { return cols[a] < cols[b]; });

        for (size_t i = 0; i < idx.size(); i++) {
            new_mat.col_idx.push_back(cols[idx[i]]);
            new_mat.values.push_back(vals[idx[i]]);
        }

        offset += cols.size();
        new_mat.row_ptr[row + 1] = offset;
    }

    new_mat.nnz = new_mat.col_idx.size();
    new_mat.avg_nnz_per_row = mat.avg_nnz_per_row;
    new_mat.std_nnz_per_row = mat.std_nnz_per_row;

    mat = std::move(new_mat);
}

// ============================================================================
// CSV Output
// ============================================================================

void writeCSVHeader(std::ofstream& file) {
    file << "timestamp,host,gpu_name,matrix_size,mode,batches,seconds_target,"
    << "seconds_gpu,seconds_wall,energy_j,avg_power_w,below_target,workload,"
    << "impl,dtype,N,passes_kernel,passes_e2e,seconds_kernel,energy_kernel_j,"
    << "avg_power_w_kernel,avg_power_w_e2e,bytes_total,bw_gb_s,time_mode,"
    << "energy_mode,includes_transfer,device_name,driver_version,"
    << "pcie_gen_current,pcie_width_current,pcie_rx_kbs,pcie_tx_kbs,"
    << "clocks_sm_mhz,clocks_mem_mhz,temp_c,throttle_reasons,notes,"
    << "rows,cols,nnz,op,sp_algo,num_rhs,backend,library,library_version,matrix_id,"
    << "pattern,permuted,band_width,block_size,avg_nnz_per_row,std_nnz_per_row\n";
}

void writeCSVRow(std::ofstream& file,
                 const std::string& timestamp,
                 const std::string& hostname,
                 const DeviceInfo& dev_info,
                 const CSRMatrix& mat,
                 const PatternSpec& spec,
                 size_t passes,
                 int repeat_idx,
                 double seconds_kernel,
                 double seconds_wall,
                 double energy_j,
                 double avg_power_w,
                 size_t bytes_total,
                 double bw_gb_s,
                 const GPURuntimeInfo& runtime,
                 int cusparse_version,
                 const std::string& extra_notes) {

    int below_target = (seconds_kernel < TARGET_S * 0.95) ? 1 : 0;

    file << timestamp << ","
    << hostname << ","
    << dev_info.name << ","
    << mat.rows << ","
    << "kernel" << ","
    << repeat_idx << ","
    << std::fixed << std::setprecision(2) << TARGET_S << ","
    << std::setprecision(4) << seconds_kernel << ","
    << seconds_wall << ","
    << std::setprecision(3) << energy_j << ","
    << std::setprecision(1) << avg_power_w << ","
    << below_target << ","
    << "spmv" << ","
    << "cusparse" << ","
    << DTYPE_STR << ","
    << mat.rows << ","
    << passes << ","
    << passes << ","
    << std::setprecision(4) << seconds_kernel << ","
    << std::setprecision(3) << energy_j << ","
    << std::setprecision(1) << avg_power_w << ","
    << avg_power_w << ","
    << bytes_total << ","
    << std::setprecision(2) << bw_gb_s << ","
    << "kernel" << ","
    << "kernel" << ","
    << 0 << ","
    << dev_info.name << ","
    << dev_info.driver_version << ","
    << runtime.pcie_gen << ","
    << runtime.pcie_width << ","
    << runtime.pcie_rx_kbs << ","
    << runtime.pcie_tx_kbs << ","
    << runtime.clocks_sm_mhz << ","
    << runtime.clocks_mem_mhz << ","
    << runtime.temp_c << ","
    << runtime.throttle_reasons << ","
    << "sweep;repeats=" << REPEATS << ";passes=" << passes
    << ";dtype=" << DTYPE_STR << ";byte_model=csr_theoretical" << extra_notes << ","
    << mat.rows << ","
    << mat.cols << ","
    << mat.nnz << ","
    << "N" << ","
    << "spmv_alg_default" << ","
    << "0" << ","
    << "cuda" << ","
    << "cusparse" << ","
    << cusparse_version << ","
    << ("synthetic:" + std::string(spec.name)) << ","
    << spec.name << ","
    << (spec.permuted ? 1 : 0) << ","
    << spec.band_width << ","
    << spec.block_size << ","
    << std::setprecision(2) << mat.avg_nnz_per_row << ","
    << std::setprecision(2) << mat.std_nnz_per_row << "\n";
                 }

                 void writeSkippedRow(std::ofstream& file,
                                      const std::string& timestamp,
                                      const std::string& hostname,
                                      const DeviceInfo& dev_info,
                                      int N,
                                      const PatternSpec& spec) {
                     file << timestamp << "," << hostname << "," << dev_info.name << ","
                     << N << ",kernel,0,0.00,0.0000,0.0000,0.000,0.0,1,"
                     << "spmv,cusparse," << DTYPE_STR << ","
                     << N << ",0,0,0.0000,0.000,0.0,0.0,0,0.00,"
                     << "kernel,kernel,0," << dev_info.name << "," << dev_info.driver_version << ","
                     << "0,0,0,0,0,0,0,0,skip_oom,"
                     << "0,0,0,N,spmv_alg_default,0,cuda,cusparse,0," << spec.name << ","
                     << spec.name << "," << (spec.permuted ? 1 : 0) << ","
                     << spec.band_width << "," << spec.block_size << ",0.00,0.00\n";
                                      }

                                      // ============================================================================
                                      // Calibration
                                      // ============================================================================

                                      size_t calibrateSpMV(cusparseHandle_t handle,
                                                           cusparseSpMatDescr_t matA,
                                                           cusparseDnVecDescr_t vecX,
                                                           cusparseDnVecDescr_t vecY,
                                                           void* buffer,
                                                           cudaStream_t stream) {
                                          const real alpha = 1.0f;
                                          const real beta = 0.0f;

                                          // Warm-up pass
                                          CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                                      &alpha, matA, vecX, &beta, vecY,
                                                                      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
                                          CUDA_CHECK(cudaStreamSynchronize(stream));

                                          // Timed calibration pass
                                          cudaEvent_t start, stop;
                                          CUDA_CHECK(cudaEventCreate(&start));
                                          CUDA_CHECK(cudaEventCreate(&stop));

                                          CUDA_CHECK(cudaEventRecord(start, stream));
                                          CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                                      &alpha, matA, vecX, &beta, vecY,
                                                                      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
                                          CUDA_CHECK(cudaEventRecord(stop, stream));
                                          CUDA_CHECK(cudaEventSynchronize(stop));

                                          float ms_single;
                                          CUDA_CHECK(cudaEventElapsedTime(&ms_single, start, stop));
                                          double t_pass = ms_single / 1000.0;

                                          CUDA_CHECK(cudaEventDestroy(start));
                                          CUDA_CHECK(cudaEventDestroy(stop));

                                          size_t passes = std::max(static_cast<size_t>(1),
                                                                   static_cast<size_t>(std::ceil((TARGET_S / t_pass) * SAFETY_FACTOR)));
                                          return passes;
                                                           }

                                                           // ============================================================================
                                                           // Main
                                                           // ============================================================================

                                                           int main() {
                                                               // Initialize NVML
                                                               NVMLContext nvml;
                                                               bool nvml_ok = nvml.init();
                                                               if (!nvml_ok) {
                                                                   std::cerr << "⚠️  NVML not available, energy measurements will be zero\n";
                                                               } else {
                                                                   std::cout << "✓ NVML initialized, energy tracking enabled\n";
                                                               }

                                                               // Get device info
                                                               DeviceInfo dev_info = getDeviceInfo();
                                                               std::string hostname = getHostname();

                                                               // Create cuSPARSE handle and stream
                                                               cusparseHandle_t handle;
                                                               CUSPARSE_CHECK(cusparseCreate(&handle));

                                                               cudaStream_t stream;
                                                               CUDA_CHECK(cudaStreamCreate(&stream));
                                                               CUSPARSE_CHECK(cusparseSetStream(handle, stream));

                                                               // Get cuSPARSE version (using cusparseGetVersion for older cuSPARSE)
                                                               int cusparse_version = 0;
                                                               cusparseGetVersion(handle, &cusparse_version);

                                                               std::cout << "========================================\n";
                                                               std::cout << "SpMV Benchmark - cuSPARSE + NVML Energy\n";
                                                               std::cout << "VERSION: 2.0.0-FIXED (2025-01-15)\n";
                                                               std::cout << "========================================\n";
                                                               std::cout << "Device:         " << dev_info.name << "\n";
                                                               std::cout << "Compute Cap:    " << dev_info.cc_major << "." << dev_info.cc_minor << "\n";
                                                               std::cout << "Driver:         " << dev_info.driver_version << "\n";
                                                               std::cout << "SMs:            " << dev_info.sm_count << "\n";
                                                               std::cout << "Total Memory:   " << readableBytes(dev_info.total_global_mem) << "\n";
                                                               std::cout << "Data type:      " << DTYPE_STR << " (" << sizeof(real) << " bytes)\n";
                                                               std::cout << "cuSPARSE ver:   " << cusparse_version << "\n";
                                                               std::cout << "Target runtime: " << TARGET_S << "s\n";
                                                               std::cout << "Repeats/size:   " << REPEATS << "\n";
                                                               std::cout << "Output:         " << CSV_PATH << "\n";
                                                               std::cout << "========================================\n\n";

                                                               // Prepare CSV
                                                               ensureDirectoryExists(CSV_PATH);
                                                               bool write_header = !fileExists(CSV_PATH);

                                                               std::ofstream csv_file(CSV_PATH, std::ios::app);
                                                               if (!csv_file.is_open()) {
                                                                   std::cerr << "Error: Cannot open " << CSV_PATH << "\n";
                                                                   return EXIT_FAILURE;
                                                               }

                                                               if (write_header) {
                                                                   writeCSVHeader(csv_file);
                                                               }

                                                               // Create CUDA events
                                                               cudaEvent_t start_event, stop_event;
                                                               CUDA_CHECK(cudaEventCreate(&start_event));
                                                               CUDA_CHECK(cudaEventCreate(&stop_event));

                                                               // ========================================================================
                                                               // PATTERN SWEEP
                                                               // ========================================================================

                                                               constexpr size_t num_sizes = sizeof(MATRIX_SIZES) / sizeof(MATRIX_SIZES[0]);
                                                               constexpr size_t num_runs = sizeof(RUNS) / sizeof(RUNS[0]);

                                                               for (size_t run_idx = 0; run_idx < num_runs; run_idx++) {
                                                                   const auto& spec = RUNS[run_idx];

                                                                   std::cout << "\n╔════════════════════════════════════════╗\n";
                                                                   std::cout << "  Pattern: " << spec.name << (spec.permuted ? " (permuted)" : "") << "\n";
                                                                   std::cout << "╚════════════════════════════════════════╝\n";

                                                                   for (size_t size_idx = 0; size_idx < num_sizes; size_idx++) {
                                                                       size_t target_rows = MATRIX_SIZES[size_idx];

                                                                       std::cout << "\n────────────────────────────────────────\n";
                                                                       std::cout << "Size " << (size_idx + 1) << "/" << num_sizes
                                                                       << ": target_rows = " << target_rows << "\n";
                                                                       std::cout << "────────────────────────────────────────\n";

                                                                       // Generate matrix based on pattern
                                                                       CSRMatrix mat;
                                                                       std::string size_notes = "";

                                                                       switch (spec.pat) {
                                                                           case STENCIL_5PT: {
                                                                               int N = static_cast<int>(std::sqrt(target_rows));
                                                                               mat = generateStencil5pt(N);
                                                                               if (mat.rows != static_cast<int>(target_rows)) {
                                                                                   size_notes = ";rows_adjusted=" + std::to_string(mat.rows) +
                                                                                   "_from_" + std::to_string(target_rows);
                                                                               }
                                                                               break;
                                                                           }
                                                                           case IRREGULAR_20: {
                                                                               int N = static_cast<int>(std::sqrt(target_rows));
                                                                               mat = generateIrregular20(N);
                                                                               if (mat.rows != static_cast<int>(target_rows)) {
                                                                                   size_notes = ";rows_adjusted=" + std::to_string(mat.rows) +
                                                                                   "_from_" + std::to_string(target_rows);
                                                                               }
                                                                               break;
                                                                           }
                                                                           case STENCIL_7PT_3D:
                                                                               mat = generateStencil7pt3D(target_rows);
                                                                               if (mat.rows != static_cast<int>(target_rows)) {
                                                                                   size_notes = ";rows_adjusted=" + std::to_string(mat.rows) +
                                                                                   "_from_" + std::to_string(target_rows);
                                                                               }
                                                                               break;
                                                                           case STENCIL_27PT_3D:
                                                                               mat = generateStencil27pt3D(target_rows);
                                                                               if (mat.rows != static_cast<int>(target_rows)) {
                                                                                   size_notes = ";rows_adjusted=" + std::to_string(mat.rows) +
                                                                                   "_from_" + std::to_string(target_rows);
                                                                               }
                                                                               break;
                                                                           case BAND_TRIDIAG:
                                                                               mat = generateBandTridiag(target_rows, spec.band_width);
                                                                               break;
                                                                           case BLOCK_DIAGONAL:
                                                                               mat = generateBlockDiagonal(target_rows, spec.block_size);
                                                                               break;
                                                                           case RMAT_POWERLAW:
                                                                               mat = generateRMAT(target_rows, spec.scale, spec.edge_factor);
                                                                               if (mat.rows != static_cast<int>(target_rows)) {
                                                                                   size_notes = ";rows_adjusted=" + std::to_string(mat.rows) +
                                                                                   "_from_" + std::to_string(target_rows) +
                                                                                   ";scale=" + std::to_string(spec.scale);
                                                                               }
                                                                               break;
                                                                       }

                                                                       // Apply permutation if requested
                                                                       if (spec.permuted) {
                                                                           permuteCSR(mat);
                                                                       }

                                                                       std::cout << "Generated: " << mat.rows << "x" << mat.cols
                                                                       << ", nnz=" << mat.nnz
                                                                       << " (avg " << std::fixed << std::setprecision(2) << mat.avg_nnz_per_row
                                                                       << " ± " << mat.std_nnz_per_row << " per row)\n";

                                                                       // Check memory
                                                                       size_t free_bytes, total_bytes;
                                                                       CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

                                                                       size_t vec_bytes = mat.cols * sizeof(real) + mat.rows * sizeof(real);
                                                                       size_t required = mat.deviceMemoryBytes() + vec_bytes;
                                                                       size_t safe_threshold = static_cast<size_t>(free_bytes * OOM_SAFETY);

                                                                       if (required > safe_threshold) {
                                                                           std::cout << "⚠️  SKIPPING: OOM (need " << readableBytes(required)
                                                                           << ", safe " << readableBytes(safe_threshold) << ")\n";
                                                                           writeSkippedRow(csv_file, getTimestamp(), hostname, dev_info,
                                                                                           mat.rows, spec);
                                                                           csv_file.flush();
                                                                           continue;
                                                                       }

                                                                       // Allocate and copy matrix
                                                                       mat.allocateDevice();
                                                                       mat.copyToDevice();

                                                                       // Allocate vectors
                                                                       real *d_x, *d_y;
                                                                       CUDA_CHECK(cudaMalloc(&d_x, mat.cols * sizeof(real)));
                                                                       CUDA_CHECK(cudaMalloc(&d_y, mat.rows * sizeof(real)));

                                                                       // Initialize x vector
                                                                       std::vector<real> h_x(mat.cols, 1.0f);
                                                                       CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), mat.cols * sizeof(real),
                                                                                             cudaMemcpyHostToDevice));
                                                                       CUDA_CHECK(cudaMemset(d_y, 0, mat.rows * sizeof(real)));

                                                                       // Create cuSPARSE descriptors
                                                                       cusparseSpMatDescr_t matA;
                                                                       cusparseDnVecDescr_t vecX, vecY;

                                                                       CUSPARSE_CHECK(cusparseCreateCsr(&matA, mat.rows, mat.cols, mat.nnz,
                                                                                                        mat.d_row_ptr, mat.d_col_idx, mat.d_values,
                                                                                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                                                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

                                                                       CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, mat.cols, d_x, CUDA_R_32F));
                                                                       CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, mat.rows, d_y, CUDA_R_32F));

                                                                       // Query buffer size
                                                                       size_t buffer_size = 0;
                                                                       const real alpha = 1.0f;
                                                                       const real beta = 0.0f;

                                                                       CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                                                                              &alpha, matA, vecX, &beta, vecY,
                                                                                                              CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                                                                                              &buffer_size));

                                                                       std::cout << "Workspace: " << readableBytes(buffer_size) << "\n";

                                                                       // Check if workspace fits in remaining VRAM
                                                                       CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                                                                       safe_threshold = static_cast<size_t>(free_bytes * OOM_SAFETY);

                                                                       if (buffer_size > safe_threshold) {
                                                                           std::cout << "⚠️  SKIPPING: Workspace OOM (" << readableBytes(buffer_size)
                                                                           << " > safe " << readableBytes(safe_threshold) << ")\n";

                                                                           // Cleanup descriptors and device memory
                                                                           CUSPARSE_CHECK(cusparseDestroySpMat(matA));
                                                                           CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
                                                                           CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
                                                                           CUDA_CHECK(cudaFree(d_x));
                                                                           CUDA_CHECK(cudaFree(d_y));
                                                                           mat.freeDevice();

                                                                           writeSkippedRow(csv_file, getTimestamp(), hostname, dev_info,
                                                                                           mat.rows, spec);
                                                                           csv_file.flush();
                                                                           continue;
                                                                       }

                                                                       void* d_buffer = nullptr;
                                                                       if (buffer_size > 0) {
                                                                           CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
                                                                       }

                                                                       // Calibrate
                                                                       std::cout << "Calibrating (with warm-up)... " << std::flush;
                                                                       size_t passes = calibrateSpMV(handle, matA, vecX, vecY, d_buffer, stream);
                                                                       std::cout << passes << " passes\n";

                                                                       // Ensure minimum passes for small matrices (to average out launch overhead)
                                                                       if (mat.rows <= 4096) {
                                                                           passes = std::max(passes, MIN_PASSES_SMALL);
                                                                       }

                                                                       // Adaptive validation: probe run with current passes
                                                                       float ms_check = 0.0f;
                                                                       {
                                                                           CUDA_CHECK(cudaEventRecord(start_event, stream));
                                                                           for (size_t p = 0; p < passes; ++p) {
                                                                               CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                                                                           &alpha, matA, vecX, &beta, vecY,
                                                                                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
                                                                           }
                                                                           CUDA_CHECK(cudaEventRecord(stop_event, stream));
                                                                           CUDA_CHECK(cudaEventSynchronize(stop_event));
                                                                           CUDA_CHECK(cudaEventElapsedTime(&ms_check, start_event, stop_event));
                                                                       }
                                                                       double seconds_check = ms_check / 1000.0;

                                                                       // One-time adjustment if below target envelope
                                                                       if (seconds_check < TARGET_LOW * TARGET_S) {
                                                                           size_t passes_new = std::max(
                                                                               passes + 1,
                                                                               static_cast<size_t>(std::ceil(passes * TARGET_S / seconds_check * 1.02))
                                                                           );
                                                                           if (passes_new > passes) {
                                                                               passes = passes_new;
                                                                               std::cout << "Adjusted passes -> " << passes
                                                                               << " (check=" << std::fixed << std::setprecision(4) << seconds_check << "s)\n";
                                                                               size_notes += ";calibration_adjusted=1;passes=" + std::to_string(passes);
                                                                           }
                                                                       }

                                                                       // Calculate bytes transferred (theoretical CSR model) - AFTER final passes
                                                                       // Per SpMV pass: nnz values, nnz col_idx, (rows+1) row_ptr, N x-vector, N y-vector
                                                                       size_t bytes_per_pass = mat.nnz * (sizeof(real) + sizeof(int)) +
                                                                       (mat.rows + 1) * sizeof(int) +
                                                                       mat.cols * sizeof(real) +
                                                                       mat.rows * sizeof(real);
                                                                       size_t bytes_total = bytes_per_pass * passes;

                                                                       // Measurement loop
                                                                       std::cout << "Measuring " << REPEATS << " runs...\n";

                                                                       for (int run = 0; run < REPEATS; run++) {
                                                                           // Reset y vector
                                                                           CUDA_CHECK(cudaMemsetAsync(d_y, 0, mat.rows * sizeof(real), stream));
                                                                           CUDA_CHECK(cudaStreamSynchronize(stream));

                                                                           // Energy start
                                                                           unsigned long long energy_start_mj = nvml.getTotalEnergyMilliJ();

                                                                           // Wall time start
                                                                           auto wall_t0 = std::chrono::steady_clock::now();

                                                                           // Timing start
                                                                           CUDA_CHECK(cudaEventRecord(start_event, stream));

                                                                           // SpMV loop
                                                                           for (size_t p = 0; p < passes; p++) {
                                                                               CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                                                                           &alpha, matA, vecX, &beta, vecY,
                                                                                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                                                                                           d_buffer));
                                                                           }

                                                                           // Timing stop
                                                                           CUDA_CHECK(cudaEventRecord(stop_event, stream));
                                                                           CUDA_CHECK(cudaEventSynchronize(stop_event));

                                                                           // Wall time stop
                                                                           auto wall_t1 = std::chrono::steady_clock::now();
                                                                           double seconds_wall = std::chrono::duration<double>(wall_t1 - wall_t0).count();

                                                                           // Energy stop
                                                                           unsigned long long energy_stop_mj = nvml.getTotalEnergyMilliJ();

                                                                           // Get runtime info
                                                                           GPURuntimeInfo runtime = getGPURuntimeInfo(nvml.device, nvml_ok);

                                                                           // Calculate metrics
                                                                           float ms = 0;
                                                                           CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
                                                                           double seconds_kernel = ms / 1000.0;

                                                                           double energy_j = 0.0;
                                                                           double avg_power_w = 0.0;
                                                                           std::string extra_notes = size_notes;

                                                                           if (nvml_ok && nvml.energy_supported && energy_stop_mj >= energy_start_mj) {
                                                                               energy_j = (energy_stop_mj - energy_start_mj) / 1000.0;
                                                                               avg_power_w = energy_j / seconds_kernel;
                                                                           } else if (!nvml_ok || !nvml.energy_supported) {
                                                                               extra_notes += ";no_energy";
                                                                           }

                                                                           double bw_gb_s = (bytes_total / 1e9) / seconds_kernel;

                                                                           // Write CSV (use seconds_wall for wall time column)
                                                                           writeCSVRow(csv_file, getTimestamp(), hostname, dev_info, mat,
                                                                                       spec, passes, run, seconds_kernel, seconds_wall,
                                                                                       energy_j, avg_power_w, bytes_total, bw_gb_s, runtime,
                                                                                       cusparse_version, extra_notes);
                                                                           csv_file.flush();

                                                                           // Console output
                                                                           if (run == 0 || run == REPEATS-1 || (run+1) % 10 == 0) {
                                                                               std::cout << "  [" << std::setw(2) << (run+1) << "/" << REPEATS << "] "
                                                                               << std::fixed << std::setprecision(4) << seconds_kernel << "s";
                                                                               if (energy_j > 0) {
                                                                                   std::cout << " E=" << std::setprecision(1) << energy_j << "J";
                                                                                   if (avg_power_w > 0) {
                                                                                       std::cout << " P=" << std::setprecision(0) << avg_power_w << "W";
                                                                                   }
                                                                               }
                                                                               std::cout << " | BW=" << std::setprecision(2) << bw_gb_s << " GB/s\n";
                                                                           }
                                                                       }

                                                                       std::cout << "✓ Complete\n";

                                                                       // Cleanup
                                                                       if (d_buffer) cudaFree(d_buffer);
                                                                       CUSPARSE_CHECK(cusparseDestroySpMat(matA));
                                                                       CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
                                                                       CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
                                                                       CUDA_CHECK(cudaFree(d_x));
                                                                       CUDA_CHECK(cudaFree(d_y));
                                                                       mat.freeDevice();
                                                                   }
                                                               }

                                                               // Final cleanup
                                                               csv_file.close();
                                                               CUDA_CHECK(cudaEventDestroy(start_event));
                                                               CUDA_CHECK(cudaEventDestroy(stop_event));
                                                               CUDA_CHECK(cudaStreamDestroy(stream));
                                                               CUSPARSE_CHECK(cusparseDestroy(handle));
                                                               nvml.shutdown();

                                                               std::cout << "\n========================================\n";
                                                               std::cout << "✓ Benchmark complete!\n";
                                                               std::cout << "Results: " << CSV_PATH << "\n";
                                                               std::cout << "========================================\n";

                                                               return EXIT_SUCCESS;
                                                           }
