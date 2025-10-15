// spmv_cpu.cpp - CSR SpMV Benchmark with OpenMP and RAPL Energy
// VERSION: 1.0.0-OpenBLAS (2025-01-15)
// Build: g++ -O3 -DNDEBUG -fopenmp -std=c++17 -march=native -o spmv_cpu spmv_cpu.cpp
//
// Usage: ./spmv_cpu --threads 8

#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>
#include <unistd.h>
#include <sys/stat.h>
#include <glob.h>
#include <cmath>

// ============================================================================
// HARDCODED CONFIGURATION
// ============================================================================

// Data type
typedef float real;
#define DTYPE_STR "fp32"

// Target measurement window
#define TARGET_S 1.0
#define TARGET_LOW 0.95
#define SAFETY_FACTOR 1.02

// Matrix sizes (total rows) - same as GPU variant
static const size_t MATRIX_SIZES[] = {
    1024, 4096, 16384, 65536, 131072, 262144
};
static const int NUM_SIZES = sizeof(MATRIX_SIZES) / sizeof(MATRIX_SIZES[0]);

// Sparsity patterns
enum SparsityPattern {
    STENCIL_5PT,
    IRREGULAR_20,
    STENCIL_7PT_3D,
    STENCIL_27PT_3D,
    BAND_TRIDIAG,
    BLOCK_DIAGONAL,
    RMAT_POWERLAW
};

struct PatternSpec {
    SparsityPattern pat;
    bool permuted;
    int band_width;
    int block_size;
    int scale;
    int edge_factor;
    const char* name;
};

// Run configurations - same as GPU
static const PatternSpec RUNS[] = {
    { STENCIL_5PT,      false, 1, 0, 0,  0,  "STENCIL_5PT_2D" },
    { IRREGULAR_20,     false, 0, 0, 0,  0,  "IRREGULAR_20" },
    { STENCIL_7PT_3D,   false, 1, 0, 0,  0,  "STENCIL_7PT_3D" },
    { STENCIL_27PT_3D,  false, 1, 0, 0,  0,  "STENCIL_27PT_3D" },
    { BAND_TRIDIAG,     false, 1, 0, 0,  0,  "BAND_TRIDIAG" },
    { BLOCK_DIAGONAL,   false, 0, 8, 0,  0,  "BLOCK_DIAGONAL" },
    { RMAT_POWERLAW,    false, 0, 0, 20, 16, "RMAT_POWERLAW" },
    { RMAT_POWERLAW,    true,  0, 0, 20, 16, "RMAT_POWERLAW_PERM" }
};
static const int NUM_RUNS = sizeof(RUNS) / sizeof(RUNS[0]);

// Seeds
#define SEED_BASE 12345

// Repeats (same as main.cu)
static constexpr int REPEATS = 50;

// CSV output
#define CSV_PATH "data/raw/spmv_cpu.csv"

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

std::string getCPUModel() {
    FILE* pipe = popen("lscpu 2>/dev/null | grep 'Model name:' | sed 's/Model name:\\s*//'", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            std::string model(buffer);
            if (!model.empty() && model.back() == '\n') {
                model.pop_back();
            }
            if (!model.empty()) return model;
        }
        pclose(pipe);
    }
    
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 1);
                model.erase(0, model.find_first_not_of(" \t"));
                return model;
            }
        }
    }
    return "Unknown CPU";
}

void ensureDirectoryExists(const char* filepath) {
    std::string path(filepath);
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path.substr(0, pos);
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
}

bool fileExists(const char* filepath) {
    struct stat buffer;
    return (stat(filepath, &buffer) == 0);
}

std::vector<std::string> globPaths(const std::string& pattern) {
    glob_t glob_result;
    std::vector<std::string> paths;
    
    if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result) == 0) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            paths.push_back(std::string(glob_result.gl_pathv[i]));
        }
    }
    globfree(&glob_result);
    return paths;
}

std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::string content;
    std::getline(file, content);
    return content;
}

// ============================================================================
// RAPL Energy Reader
// ============================================================================

struct RaplSnap {
    unsigned long long e_uj;
    unsigned long long range_uj;
};

struct RaplReader {
    std::string energy_path;
    std::string range_path;
    bool valid;
};

RaplReader initRapl() {
    RaplReader rapl;
    rapl.valid = false;
    
    // Find Package domain
    auto base_zones = globPaths("/sys/class/powercap/*rapl*");
    
    for (const auto& zone : base_zones) {
        struct stat st;
        if (stat(zone.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;
        
        std::string name_path = zone + "/name";
        std::string name = readFile(name_path);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        
        if (name.find("package") != std::string::npos) {
            rapl.energy_path = zone + "/energy_uj";
            rapl.range_path = zone + "/max_energy_range_uj";
            
            // Test read
            std::ifstream test(rapl.energy_path);
            if (test.good()) {
                rapl.valid = true;
                break;
            }
        }
    }
    
    return rapl;
}

RaplSnap readRapl(const RaplReader& rapl) {
    RaplSnap snap = {0, 0};
    if (!rapl.valid) return snap;
    
    std::ifstream e_file(rapl.energy_path);
    std::ifstream r_file(rapl.range_path);
    
    e_file >> snap.e_uj;
    r_file >> snap.range_uj;
    
    return snap;
}

double deltaJoules(const RaplSnap& before, const RaplSnap& after) {
    unsigned long long delta_uj;
    
    if (after.e_uj >= before.e_uj) {
        delta_uj = after.e_uj - before.e_uj;
    } else {
        // Overflow
        delta_uj = after.e_uj + (before.range_uj - before.e_uj);
    }
    
    return delta_uj * 1e-6;  // µJ → J
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
    
    double avg_nnz_per_row;
    double std_nnz_per_row;
};

void computeMatrixStatistics(CSRMatrix& mat) {
    if (mat.rows == 0) return;
    
    std::vector<int> nnz_per_row(mat.rows);
    for (int i = 0; i < mat.rows; i++) {
        nnz_per_row[i] = mat.row_ptr[i+1] - mat.row_ptr[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < mat.rows; i++) {
        sum += nnz_per_row[i];
    }
    mat.avg_nnz_per_row = sum / mat.rows;
    
    double var_sum = 0.0;
    for (int i = 0; i < mat.rows; i++) {
        double diff = nnz_per_row[i] - mat.avg_nnz_per_row;
        var_sum += diff * diff;
    }
    mat.std_nnz_per_row = std::sqrt(var_sum / mat.rows);
}

// ============================================================================
// CSR SpMV Kernel (OpenMP parallelized)
// ============================================================================

void csr_spmv(const CSRMatrix& mat, const real* x, real* y, real alpha, real beta) {
    // y = alpha * A * x + beta * y
    
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < mat.rows; row++) {
        real sum = 0.0;
        
        for (int p = mat.row_ptr[row]; p < mat.row_ptr[row + 1]; p++) {
            sum += mat.values[p] * x[mat.col_idx[p]];
        }
        
        if (beta == 0.0) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

// ============================================================================
// Matrix Generators
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
    computeMatrixStatistics(mat);
    return mat;
}

CSRMatrix generateIrregular20(int N, uint32_t seed) {
    CSRMatrix mat;
    mat.rows = N * N;
    mat.cols = N * N;
    mat.row_ptr.resize(mat.rows + 1);
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nnz_dist(15, 25);
    std::uniform_int_distribution<int> col_dist(0, mat.cols - 1);
    std::uniform_real_distribution<real> val_dist(0.1f, 1.0f);
    
    int offset = 0;
    mat.row_ptr[0] = 0;
    
    for (int row = 0; row < mat.rows; row++) {
        int nnz_row = std::min(nnz_dist(rng), mat.cols);
        std::vector<int> cols;
        
        cols.push_back(row); // Diagonal
        
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
    computeMatrixStatistics(mat);
    return mat;
}

CSRMatrix generateStencil7pt3D(int total_rows) {
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
                
                temp_cols[row].push_back(row);
                temp_vals[row].push_back(6.0f);
                
                if (i > 0) {
                    temp_cols[row].push_back(idx(i-1, j, k));
                    temp_vals[row].push_back(-1.0f);
                }
                if (i < n-1) {
                    temp_cols[row].push_back(idx(i+1, j, k));
                    temp_vals[row].push_back(-1.0f);
                }
                if (j > 0) {
                    temp_cols[row].push_back(idx(i, j-1, k));
                    temp_vals[row].push_back(-1.0f);
                }
                if (j < n-1) {
                    temp_cols[row].push_back(idx(i, j+1, k));
                    temp_vals[row].push_back(-1.0f);
                }
                if (k > 0) {
                    temp_cols[row].push_back(idx(i, j, k-1));
                    temp_vals[row].push_back(-1.0f);
                }
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
    computeMatrixStatistics(mat);
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
    computeMatrixStatistics(mat);
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
    computeMatrixStatistics(mat);
    return mat;
}

CSRMatrix generateBlockDiagonal(int total_rows, int block_size, uint32_t seed) {
    CSRMatrix mat;
    mat.rows = total_rows;
    mat.cols = total_rows;
    mat.row_ptr.resize(mat.rows + 1);
    
    std::mt19937 rng(seed);
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
    computeMatrixStatistics(mat);
    return mat;
}

CSRMatrix generateRMAT(int total_rows, int scale, int edge_factor, uint32_t seed) {
    CSRMatrix mat;
    
    int n_bits = (scale > 0) ? scale : static_cast<int>(std::ceil(std::log2(total_rows)));
    int n = 1 << n_bits;
    
    mat.rows = n;
    mat.cols = n;
    
    const double a = 0.57, b = 0.19, c = 0.19;
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_real_distribution<real> val_dist(0.1f, 1.0f);
    
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
            edges.push_back({v, u});
        }
    }
    
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
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
    computeMatrixStatistics(mat);
    return mat;
}

void permuteCSR(CSRMatrix& mat, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int> perm(mat.rows);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    
    std::vector<int> inv_perm(mat.rows);
    for (int i = 0; i < mat.rows; i++) {
        inv_perm[perm[i]] = i;
    }
    
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
    file << "timestamp,host,cpu_model,backend,library,library_version,impl,workload,dtype,"
         << "pattern,nrows,ncols,nnz,num_threads,batches,"
         << "seconds_target,passes_kernel,seconds_kernel,"
         << "energy_j,avg_power_w,below_target,bytes_total,bw_gb_s,time_mode,energy_mode,"
         << "omp_proc_bind,omp_places,energy_domain,numa_policy,notes\n";
}

void writeCSVRow(std::ofstream& file,
                 const std::string& timestamp,
                 const std::string& hostname,
                 const std::string& cpu_model,
                 const std::string& library_version,
                 const CSRMatrix& mat,
                 const PatternSpec& spec,
                 int num_threads,
                 int batches,
                 size_t passes,
                 double seconds_kernel,
                 double energy_j,
                 double avg_power_w,
                 size_t bytes_total,
                 double bw_gb_s,
                 const std::string& notes) {
    
    int below_target = (seconds_kernel < TARGET_S * 0.95) ? 1 : 0;
    
    file << timestamp << ","
         << hostname << ","
         << cpu_model << ","
         << "cpu" << ","
         << "OpenMP" << ","
         << library_version << ","
         << "omp" << ","
         << "spmv" << ","
         << DTYPE_STR << ","
         << spec.name << ","
         << mat.rows << ","
         << mat.cols << ","
         << mat.nnz << ","
         << num_threads << ","
         << batches << ","
         << std::fixed << std::setprecision(2) << TARGET_S << ","
         << passes << ","
         << std::setprecision(4) << seconds_kernel << ",";
    
    if (energy_j >= 0) {
        file << std::setprecision(3) << energy_j;
    } else {
        file << "NA";
    }
    file << ",";
    
    if (avg_power_w >= 0) {
        file << std::setprecision(1) << avg_power_w;
    } else {
        file << "NA";
    }
    file << ",";
    
    file << below_target << ","
         << bytes_total << ","
         << std::setprecision(2) << bw_gb_s << ","
         << "kernel" << ","
         << "kernel" << ","
         << "spread" << ","
         << "cores" << ","
         << "rapl_pkg" << ","
         << "first_touch" << ","
         << "byte_model=csr_theoretical;hint=off;" << notes << "\n";
}

// ============================================================================
// Bytes Model
// ============================================================================

uint64_t bytes_per_pass_csr(const CSRMatrix& mat) {
    uint64_t nnz_bytes = mat.nnz * (sizeof(real) + sizeof(int));
    uint64_t rowptr_bytes = (mat.rows + 1) * sizeof(int);
    uint64_t x_bytes = mat.cols * sizeof(real);
    uint64_t y_bytes = 2 * mat.rows * sizeof(real);
    
    return nnz_bytes + rowptr_bytes + x_bytes + y_bytes;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " --threads <N>\n";
    std::cerr << "   or: " << prog << " -t <N>\n";
}

int main(int argc, char** argv) {
    // Parse CLI
    if (argc != 3) {
        printUsage(argv[0]);
        return 2;
    }
    
    int num_threads = -1;
    std::string arg1(argv[1]);
    if (arg1 == "--threads" || arg1 == "-t") {
        num_threads = std::atoi(argv[2]);
    } else {
        printUsage(argv[0]);
        return 2;
    }
    
    if (num_threads <= 0) {
        std::cerr << "Error: Invalid thread count\n";
        return 2;
    }
    
    // Set environment before OpenMP init
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "cores", 1);
    
    // Configure OpenMP
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);
    
    // Get system info
    std::string hostname = getHostname();
    std::string cpu_model = getCPUModel();
    std::string library_version = "native-omp";
    
    // Initialize RAPL
    RaplReader rapl = initRapl();
    if (!rapl.valid) {
        std::cerr << "WARN: RAPL not accessible. Energy will be NA.\n";
    }
    
    // Print configuration
    std::cout << "========================================\n";
    std::cout << "CPU SpMV Benchmark - OpenMP + RAPL\n";
    std::cout << "VERSION: 1.0.0-OpenBLAS (2025-01-15)\n";
    std::cout << "========================================\n";
    std::cout << "System:         " << hostname << "\n";
    std::cout << "CPU:            " << cpu_model << "\n";
    std::cout << "Backend:        Native OpenMP SpMV\n";
    std::cout << "Library:        " << library_version << "\n";
    std::cout << "Threads:        " << num_threads << "\n";
    std::cout << "OMP_PROC_BIND:  spread\n";
    std::cout << "OMP_PLACES:     cores\n";
    std::cout << "Data type:      " << DTYPE_STR << "\n";
    std::cout << "Target runtime: " << TARGET_S << "s\n";
    std::cout << "Repeats:        " << REPEATS << "\n";
    std::cout << "RAPL available: " << (rapl.valid ? "Yes" : "No") << "\n";
    std::cout << "Output:         " << CSV_PATH << "\n";
    std::cout << "========================================\n\n";
    
    // Prepare CSV
    ensureDirectoryExists(CSV_PATH);
    bool write_header = !fileExists(CSV_PATH);
    
    std::ofstream csv_file(CSV_PATH, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open " << CSV_PATH << "\n";
        return 1;
    }
    
    if (write_header) {
        writeCSVHeader(csv_file);
    }
    
    // ========================================================================
    // PATTERN SWEEP
    // ========================================================================
    
    for (int run_idx = 0; run_idx < NUM_RUNS; run_idx++) {
        const auto& spec = RUNS[run_idx];
        
        std::cout << "\n╔════════════════════════════════════════╗\n";
        std::cout << "  Pattern: " << spec.name << (spec.permuted ? " (permuted)" : "") << "\n";
        std::cout << "╚════════════════════════════════════════╝\n";
        
        for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
            size_t target_rows = MATRIX_SIZES[size_idx];
            
            // Compute deterministic seed
            uint32_t seed = static_cast<uint32_t>(SEED_BASE + 131u * run_idx + 10007u * size_idx);
            
            std::cout << "\n────────────────────────────────────────\n";
            std::cout << "Size " << (size_idx + 1) << "/" << NUM_SIZES
                      << ": target_rows = " << target_rows << "\n";
            std::cout << "────────────────────────────────────────\n";
            
            // Generate matrix
            CSRMatrix mat;
            std::string size_notes = "";
            
            switch (spec.pat) {
                case STENCIL_5PT: {
                    int N = static_cast<int>(std::sqrt(target_rows));
                    mat = generateStencil5pt(N);
                    if (mat.rows != static_cast<int>(target_rows)) {
                        size_notes = "rows_adjusted=" + std::to_string(mat.rows) +
                                   "_from_" + std::to_string(target_rows);
                    }
                    break;
                }
                case IRREGULAR_20: {
                    int N = static_cast<int>(std::sqrt(target_rows));
                    mat = generateIrregular20(N, seed);
                    if (mat.rows != static_cast<int>(target_rows)) {
                        size_notes = "rows_adjusted=" + std::to_string(mat.rows) +
                                   "_from_" + std::to_string(target_rows);
                    }
                    break;
                }
                case STENCIL_7PT_3D:
                    mat = generateStencil7pt3D(target_rows);
                    if (mat.rows != static_cast<int>(target_rows)) {
                        size_notes = "rows_adjusted=" + std::to_string(mat.rows) +
                                   "_from_" + std::to_string(target_rows);
                    }
                    break;
                case STENCIL_27PT_3D:
                    mat = generateStencil27pt3D(target_rows);
                    if (mat.rows != static_cast<int>(target_rows)) {
                        size_notes = "rows_adjusted=" + std::to_string(mat.rows) +
                                   "_from_" + std::to_string(target_rows);
                    }
                    break;
                case BAND_TRIDIAG:
                    mat = generateBandTridiag(target_rows, spec.band_width);
                    break;
                case BLOCK_DIAGONAL:
                    mat = generateBlockDiagonal(target_rows, spec.block_size, seed);
                    break;
                case RMAT_POWERLAW:
                    mat = generateRMAT(target_rows, spec.scale, spec.edge_factor, seed);
                    if (mat.rows != static_cast<int>(target_rows)) {
                        size_notes = "rows_adjusted=" + std::to_string(mat.rows) +
                                   "_from_" + std::to_string(target_rows) +
                                   ";scale=" + std::to_string(spec.scale);
                    }
                    break;
            }
            
            if (spec.permuted) {
                permuteCSR(mat, seed + 997);
            }
            
            std::cout << "Generated: " << mat.rows << "x" << mat.cols
                      << ", nnz=" << mat.nnz
                      << " (avg " << std::fixed << std::setprecision(2) << mat.avg_nnz_per_row
                      << " ± " << mat.std_nnz_per_row << " per row)\n";
            
            // First-touch for CSR arrays (NUMA)
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < mat.rows + 1; ++i) {
                volatile int s = mat.row_ptr[i]; (void)s;
            }
            
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < mat.nnz; ++p) {
                volatile int c = mat.col_idx[p];
                volatile real      v = mat.values[p];
                (void)c; (void)v;
            }
            
            // Allocate vectors (first-touch)
            std::vector<real> x(mat.cols);
            std::vector<real> y(mat.rows);
            
            // First-touch initialization
            #pragma omp parallel for
            for (int i = 0; i < mat.cols; i++) {
                x[i] = 1.0f;
            }
            #pragma omp parallel for
            for (int i = 0; i < mat.rows; i++) {
                y[i] = 0.0f;
            }
            
            const real alpha = 1.0f;
            const real beta = 0.0f;
            
            // Warm-up
            csr_spmv(mat, x.data(), y.data(), alpha, beta);
            
            // Calibrate (once per pattern×size)
            std::cout << "Calibrating... " << std::flush;
            
            auto t0 = std::chrono::steady_clock::now();
            csr_spmv(mat, x.data(), y.data(), alpha, beta);
            auto t1 = std::chrono::steady_clock::now();
            
            double t_first = std::chrono::duration<double>(t1 - t0).count();
            size_t passes = std::max(static_cast<size_t>(1),
                                    static_cast<size_t>(std::ceil(TARGET_S / t_first * SAFETY_FACTOR)));
            
            std::cout << passes << " passes\n";
            
            // Probe run
            auto t_probe_start = std::chrono::steady_clock::now();
            for (size_t p = 0; p < passes; p++) {
                csr_spmv(mat, x.data(), y.data(), alpha, beta);
            }
            auto t_probe_end = std::chrono::steady_clock::now();
            double seconds_probe = std::chrono::duration<double>(t_probe_end - t_probe_start).count();
            
            // One-time adjustment
            if (seconds_probe < TARGET_LOW * TARGET_S) {
                size_t passes_new = std::max(
                    passes + 1,
                    static_cast<size_t>(std::ceil(passes * TARGET_S / seconds_probe * 1.02))
                );
                if (passes_new > passes) {
                    passes = passes_new;
                    std::cout << "Adjusted passes -> " << passes
                              << " (probe=" << std::fixed << std::setprecision(4) << seconds_probe << "s)\n";
                    if (!size_notes.empty()) size_notes += ";";
                    size_notes += "calibration_adjusted=1;passes=" + std::to_string(passes);
                }
            }
            
            // Calculate bytes
            uint64_t bytes_per_pass = bytes_per_pass_csr(mat);
            uint64_t bytes_total = bytes_per_pass * passes;
            
            // Measurement loop: REPEATS times
            std::cout << "Measuring " << REPEATS << " runs...\n";
            
            for (int repeat = 0; repeat < REPEATS; repeat++) {
                // Reset y
                #pragma omp parallel for
                for (int i = 0; i < mat.rows; i++) {
                    y[i] = 0.0f;
                }
                
                RaplSnap snap0 = readRapl(rapl);
                auto wall_t0 = std::chrono::steady_clock::now();
                
                for (size_t p = 0; p < passes; p++) {
                    csr_spmv(mat, x.data(), y.data(), alpha, beta);
                }
                
                auto wall_t1 = std::chrono::steady_clock::now();
                RaplSnap snap1 = readRapl(rapl);
                
                double seconds_kernel = std::chrono::duration<double>(wall_t1 - wall_t0).count();
                
                double energy_j = rapl.valid ? deltaJoules(snap0, snap1) : -1.0;
                double avg_power_w = (energy_j >= 0.0) ? (energy_j / seconds_kernel) : -1.0;
                
                double bw_gb_s = (bytes_total / 1e9) / seconds_kernel;
                
                // Write CSV
                writeCSVRow(csv_file, getTimestamp(), hostname, cpu_model, library_version,
                           mat, spec, num_threads, repeat, passes, seconds_kernel,
                           energy_j, avg_power_w, bytes_total, bw_gb_s, size_notes);
                csv_file.flush();
                
                // Console output
                if (repeat == 0 || repeat == REPEATS-1 || (repeat+1) % 10 == 0) {
                    std::cout << "  [" << std::setw(2) << (repeat+1) << "/" << REPEATS << "] "
                              << std::fixed << std::setprecision(4) << seconds_kernel << "s";
                    if (energy_j >= 0) {
                        std::cout << " | E=" << std::setprecision(1) << energy_j << "J";
                        if (avg_power_w >= 0) {
                            std::cout << " P=" << std::setprecision(0) << avg_power_w << "W";
                        }
                    }
                    std::cout << " | BW=" << std::setprecision(2) << bw_gb_s << " GB/s\n";
                }
            }
            
            std::cout << "✓ Complete\n";
        }
    }
    
    csv_file.close();
    
    std::cout << "\n========================================\n";
    std::cout << "✓ Benchmark complete!\n";
    std::cout << "Results: " << CSV_PATH << "\n";
    std::cout << "========================================\n";
    
    return 0;
}