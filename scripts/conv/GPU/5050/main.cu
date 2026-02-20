// ============================================================================
// GPU Conv2D Energy Benchmark for RTX 5050 with cuDNN
// Operation: 2D Convolution (NCHW layout)
// ============================================================================
// Compile: nvcc -O3 -std=c++17 -o conv2d_5050 main.cu -lcudnn -lnvidia-ml
// Usage: ./conv2d_5050 [--test|-t] [--output|-o <path>] [--device|-d <id>]

#include <cuda_runtime.h>
#include <cudnn.h>
#include <nvml.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <ctime>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <filesystem>
#include <locale>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

// ============================================================================
// Configuration
// ============================================================================

constexpr double TARGET_RUNTIME_S = 1.0;
constexpr int    MAX_BATCH_SIZE   = 100000;
constexpr int    MACRO_REPEATS    = 50;

// ============================================================================
// Conv2D Shape Definition
// ============================================================================

struct Conv2DShape {
    int shape_id;
    int N, C, H, W;       // Input: batch, channels, height, width
    int K, R, S;          // Kernel: output channels, height, width
    int stride, pad;
    int H_out, W_out;
    double flops_per_batch;
    
    void compute_derived() {
        H_out = (H + 2 * pad - R) / stride + 1;
        W_out = (W + 2 * pad - S) / stride + 1;
        flops_per_batch = 2.0 * N * K * C * R * S * H_out * W_out;
    }
};

static Conv2DShape CONV_SHAPES[] = {
    {1, 32, 64,  56,  56,  64,  3, 3, 1, 1, 0, 0, 0},  // ResNet conv3_x
    {2, 32, 64,  56,  56,  128, 3, 3, 2, 1, 0, 0, 0},  // Downsample
    {3, 32, 128, 28,  28,  256, 3, 3, 2, 1, 0, 0, 0},  // conv4_x entry
    {4, 32, 256, 14,  14,  512, 3, 3, 2, 1, 0, 0, 0},  // conv5_x entry
    {5, 32, 3,   224, 224, 64,  7, 7, 2, 3, 0, 0, 0},  // Stem (large spatial)
    {6, 32, 256, 56,  56,  256, 1, 1, 1, 0, 0, 0, 0},  // Pointwise 1x1
};
static const int NUM_SHAPES = sizeof(CONV_SHAPES) / sizeof(CONV_SHAPES[0]);

// ============================================================================
// Error Checking Macros
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NVML(call) do { \
    nvmlReturn_t ret = call; \
    if (ret != NVML_SUCCESS) { \
        fprintf(stderr, "NVML Error at %s:%d: %s\n", __FILE__, __LINE__, nvmlErrorString(ret)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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

void ensureDirectoryExists(const char* filepath) {
    namespace fs = std::filesystem;
    fs::path file_path(filepath);
    if (file_path.has_parent_path()) fs::create_directories(file_path.parent_path());
}

bool fileExists(const char* filepath) {
    struct stat buffer;
    return (stat(filepath, &buffer) == 0);
}

void initializeBuffer(float* buf, size_t size, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) buf[i] = dist(gen);
}

// ============================================================================
// NVML Helper Functions
// ============================================================================

struct GPUTelemetry {
    unsigned int pcie_gen, pcie_width, sm_clock, mem_clock, temp;
    unsigned long long throttle_reasons;
};

std::string getGPUName(nvmlDevice_t device) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
    std::string full_name(name);
    const char* prefixes[] = {"NVIDIA GeForce ", "NVIDIA Tesla ", "NVIDIA "};
    for (const char* prefix : prefixes) {
        if (full_name.find(prefix) == 0) return full_name.substr(strlen(prefix));
    }
    return full_name;
}

unsigned long long getGPUEnergy(nvmlDevice_t device) {
    unsigned long long energy_mj = 0;
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
    return energy_mj;
}

GPUTelemetry getGPUTelemetry(nvmlDevice_t device) {
    GPUTelemetry t;
    CHECK_NVML(nvmlDeviceGetCurrPcieLinkGeneration(device, &t.pcie_gen));
    CHECK_NVML(nvmlDeviceGetCurrPcieLinkWidth(device, &t.pcie_width));
    CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &t.sm_clock));
    CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &t.mem_clock));
    CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &t.temp));
    if (nvmlDeviceGetCurrentClocksThrottleReasons(device, &t.throttle_reasons) != NVML_SUCCESS)
        t.throttle_reasons = 0;
    return t;
}

// ============================================================================
// cuDNN Conv2D Wrapper
// ============================================================================

class CuDNNConv2D {
public:
    CuDNNConv2D() : handle_(nullptr), workspace_(nullptr), workspace_size_(0), initialized_(false) {
        CHECK_CUDNN(cudnnCreate(&handle_));
    }
    
    ~CuDNNConv2D() {
        cleanup();
        if (handle_) cudnnDestroy(handle_);
    }
    
    void cleanup() {
        if (initialized_) {
            if (workspace_) { cudaFree(workspace_); workspace_ = nullptr; }
            cudnnDestroyTensorDescriptor(input_desc_);
            cudnnDestroyTensorDescriptor(output_desc_);
            cudnnDestroyFilterDescriptor(filter_desc_);
            cudnnDestroyConvolutionDescriptor(conv_desc_);
            initialized_ = false;
        }
    }
    
    void setup(const Conv2DShape& shape) {
        // Clean up previous setup if any
        cleanup();
        
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
        
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                shape.N, shape.C, shape.H, shape.W));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                shape.K, shape.C, shape.R, shape.S));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_, shape.pad, shape.pad,
                                                     shape.stride, shape.stride, 1, 1,
                                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                shape.N, shape.K, shape.H_out, shape.W_out));
        
        // Find best algorithm
        int algo_count;
        cudnnConvolutionFwdAlgoPerf_t perf_results[10];
        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(handle_, input_desc_, filter_desc_,
                                                          conv_desc_, output_desc_, 10, &algo_count, perf_results));
        algo_ = perf_results[0].algo;
        
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_, input_desc_, filter_desc_,
                                                            conv_desc_, output_desc_, algo_, &workspace_size_));
        if (workspace_size_ > 0) {
            CHECK_CUDA(cudaMalloc(&workspace_, workspace_size_));
        }
        
        output_size_ = (size_t)shape.N * shape.K * shape.H_out * shape.W_out;
        initialized_ = true;
    }
    
    void execute(float* d_input, float* d_filter, float* d_output, cudaStream_t stream) {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN(cudnnSetStream(handle_, stream));
        CHECK_CUDNN(cudnnConvolutionForward(handle_, &alpha, input_desc_, d_input,
                                            filter_desc_, d_filter, conv_desc_, algo_,
                                            workspace_, workspace_size_, &beta, output_desc_, d_output));
    }
    
    size_t output_size() const { return output_size_; }
    
private:
    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnConvolutionFwdAlgo_t algo_;
    void* workspace_;
    size_t workspace_size_;
    size_t output_size_;
    bool initialized_;
};

// ============================================================================
// Checksum Kernel for DCE Prevention
// ============================================================================

__global__ void reduce_sum_kernel(const float* data, float* result, size_t n) {
    __shared__ float sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (i < n) {
        sum += data[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// ============================================================================
// Batch Determination
// ============================================================================

struct BatchResult { int batches; bool below_target; };

BatchResult determineBatchSize(CuDNNConv2D& conv, float* d_input, float* d_filter, float* d_output,
                               float* d_checksum, float target_seconds, cudaStream_t stream) {
    int batch = 1;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    while (batch <= MAX_BATCH_SIZE) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int b = 0; b < batch; b++) {
            conv.execute(d_input, d_filter, d_output, stream);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float elapsed = ms / 1000.0f;
        
        if (elapsed >= target_seconds) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batch, false};
        }
        if (batch >= MAX_BATCH_SIZE) {
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            return {batch, true};
        }
        batch = std::min(batch * 2, MAX_BATCH_SIZE);
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return {batch, false};
}

// ============================================================================
// CSV Output
// ============================================================================

void writeCSVHeader(std::ofstream& csv) {
    csv << "timestamp,run_id_global,run_id_per_size,device_name,num_threads,problem_size,"
        << "batches,gpu_e2e_time_s,gpu_kernel_time_s,wall_time_s,total_energy_j,"
        << "energy_per_batch_j,energy_per_second_j,energy_per_flop_j,"
        << "time_per_gemm_ms_kernel,time_per_gemm_ms_e2e,flops_total,gflops_per_s,"
        << "avg_power_w,below_target,pcie_gen,pcie_width,sm_clock_mhz,mem_clock_mhz,"
        << "temp_c,throttle_reasons\n";
}

void writeCSVRow(std::ofstream& csv, int run_id_global, int run_id_per_size,
                 const std::string& device_name, int shape_id, int batches,
                 float gpu_e2e_time, float gpu_kernel_time, float wall_time,
                 double total_energy, double avg_power, bool below_target,
                 double flops_per_batch, const GPUTelemetry& telem) {
    
    double flops_total = flops_per_batch * batches;
    double energy_per_batch = (batches > 0) ? (total_energy / batches) : 0.0;
    double energy_per_second = (wall_time > 0) ? (total_energy / wall_time) : 0.0;
    double energy_per_flop = (flops_total > 0) ? (total_energy / flops_total) : 0.0;
    double time_per_kernel_ms = (batches > 0) ? (1e3 * gpu_kernel_time / batches) : 0.0;
    double time_per_e2e_ms = (batches > 0) ? (1e3 * gpu_e2e_time / batches) : 0.0;
    double gflops_per_s = (gpu_kernel_time > 0) ? (flops_total / gpu_kernel_time / 1e9) : 0.0;
    
    csv.imbue(std::locale::classic());
    csv << getTimestamp() << "," << run_id_global << "," << run_id_per_size << ","
        << device_name << ",," << shape_id << "," << batches << ","
        << std::fixed << std::setprecision(6)
        << gpu_e2e_time << "," << gpu_kernel_time << "," << wall_time << "," << total_energy << ","
        << std::scientific << std::setprecision(6) << energy_per_batch << ","
        << std::fixed << std::setprecision(6) << energy_per_second << ","
        << std::scientific << std::setprecision(6) << energy_per_flop << ","
        << std::fixed << std::setprecision(6) << time_per_kernel_ms << "," << time_per_e2e_ms << ","
        << std::scientific << std::setprecision(6) << flops_total << ","
        << std::fixed << std::setprecision(2) << gflops_per_s << "," << avg_power << ","
        << (below_target ? "t" : "f") << ","
        << telem.pcie_gen << "," << telem.pcie_width << "," << telem.sm_clock << ","
        << telem.mem_clock << "," << telem.temp << ",0x" << std::hex << telem.throttle_reasons << std::dec << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    bool test_mode = false;
    int device_id = 0;
    std::string output_file = "data/raw/Conv2D_5050.csv";
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--test") || !strcmp(argv[i], "-t")) test_mode = true;
        else if ((!strcmp(argv[i], "--output") || !strcmp(argv[i], "-o")) && i + 1 < argc) output_file = argv[++i];
        else if ((!strcmp(argv[i], "--device") || !strcmp(argv[i], "-d")) && i + 1 < argc) device_id = atoi(argv[++i]);
    }
    
    std::cout << "\nCUDA Conv2D Energy Benchmark (RTX 5050)\n";
    if (test_mode) std::cout << "Test mode enabled\n";
    
    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "Device: " << prop.name << " (" << (prop.totalGlobalMem / 1024 / 1024 / 1024) << " GB)\n";
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    nvmlDevice_t nvml_device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
    std::string device_name = getGPUName(nvml_device);
    
    // Compute derived shape values
    for (int i = 0; i < NUM_SHAPES; i++) {
        CONV_SHAPES[i].compute_derived();
        std::cout << "Shape " << CONV_SHAPES[i].shape_id << ": "
                  << CONV_SHAPES[i].N << "x" << CONV_SHAPES[i].C << "x"
                  << CONV_SHAPES[i].H << "x" << CONV_SHAPES[i].W << " -> "
                  << CONV_SHAPES[i].N << "x" << CONV_SHAPES[i].K << "x"
                  << CONV_SHAPES[i].H_out << "x" << CONV_SHAPES[i].W_out
                  << " | FLOPs=" << std::scientific << CONV_SHAPES[i].flops_per_batch << "\n";
    }
    
    // Find max buffer sizes
    size_t max_input = 0, max_filter = 0, max_output = 0;
    for (int i = 0; i < NUM_SHAPES; i++) {
        max_input = std::max(max_input, (size_t)CONV_SHAPES[i].N * CONV_SHAPES[i].C * CONV_SHAPES[i].H * CONV_SHAPES[i].W);
        max_filter = std::max(max_filter, (size_t)CONV_SHAPES[i].K * CONV_SHAPES[i].C * CONV_SHAPES[i].R * CONV_SHAPES[i].S);
        max_output = std::max(max_output, (size_t)CONV_SHAPES[i].N * CONV_SHAPES[i].K * CONV_SHAPES[i].H_out * CONV_SHAPES[i].W_out);
    }
    
    // Allocate host memory (no pinned!)
    float *h_input = (float*)malloc(max_input * sizeof(float));
    float *h_filter = (float*)malloc(max_filter * sizeof(float));
    float *h_output = (float*)malloc(max_output * sizeof(float));
    
    if (!h_input || !h_filter || !h_output) {
        std::cerr << "Host allocation failed\n";
        return EXIT_FAILURE;
    }
    
    initializeBuffer(h_input, max_input, 42);
    initializeBuffer(h_filter, max_filter, 43);
    
    // Allocate device memory
    float *d_input, *d_filter, *d_output, *d_checksum;
    CHECK_CUDA(cudaMalloc(&d_input, max_input * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter, max_filter * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, max_output * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_checksum, sizeof(float)));
    
    std::cout << "Allocated: input=" << (max_input * 4 / 1024 / 1024) << "MB, "
              << "filter=" << (max_filter * 4 / 1024 / 1024) << "MB, "
              << "output=" << (max_output * 4 / 1024 / 1024) << "MB\n\n";
    
    // Prepare CSV
    ensureDirectoryExists(output_file.c_str());
    bool write_header = !fileExists(output_file.c_str());
    std::ofstream csv_file(output_file, std::ios::app);
    if (write_header) writeCSVHeader(csv_file);
    
    // Create stream and events
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start_event, stop_event, start_kernel, stop_kernel;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    
    std::vector<int> shapes_to_test;
    if (test_mode) shapes_to_test = {1, 2};
    else for (int i = 0; i < NUM_SHAPES; i++) shapes_to_test.push_back(i + 1);
    
    int run_id_global = 1;
    int total_rows = 0;
    
    for (int shape_id : shapes_to_test) {
        Conv2DShape& shape = CONV_SHAPES[shape_id - 1];
        std::cout << "=== Shape " << shape_id << " ===\n";
        
        size_t input_size = (size_t)shape.N * shape.C * shape.H * shape.W;
        size_t filter_size = (size_t)shape.K * shape.C * shape.R * shape.S;
        size_t output_size = (size_t)shape.N * shape.K * shape.H_out * shape.W_out;
        
        // Setup cuDNN for this shape
        CuDNNConv2D conv;
        conv.setup(shape);
        
        // Transfer filter (constant across runs)
        CHECK_CUDA(cudaMemcpyAsync(d_filter, h_filter, filter_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Determine batch size
        std::cout << "  Determining batch size..." << std::flush;
        BatchResult batch_result = determineBatchSize(conv, d_input, d_filter, d_output, d_checksum, TARGET_RUNTIME_S, stream);
        int batches = batch_result.batches;
        std::cout << " " << batches << " batches\n";
        
        int run_id_per_size = 1;
        
        for (int rep = 0; rep < MACRO_REPEATS; rep++) {
            // E2E Measurement
            auto wall_start = std::chrono::steady_clock::now();
            unsigned long long energy_before = getGPUEnergy(nvml_device);
            
            CHECK_CUDA(cudaEventRecord(start_event, stream));
            
            // H2D
            CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(d_filter, h_filter, filter_size * sizeof(float), cudaMemcpyHostToDevice, stream));
            
            CHECK_CUDA(cudaEventRecord(start_kernel, stream));
            
            // Compute (ONLY Conv2D in kernel timing window)
            for (int b = 0; b < batches; b++) {
                conv.execute(d_input, d_filter, d_output, stream);
            }
            
            CHECK_CUDA(cudaEventRecord(stop_kernel, stream));
            
            // DCE prevention AFTER kernel timing (counts towards E2E, not kernel)
            // Note: cuDNN calls won't be optimized away, but we keep this for consistency
            CHECK_CUDA(cudaMemsetAsync(d_checksum, 0, sizeof(float), stream));
            int blocks = std::min((int)((output_size + 255) / 256), 1024);
            reduce_sum_kernel<<<blocks, 256, 0, stream>>>(d_output, d_checksum, output_size);
            
            // D2H
            CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
            
            CHECK_CUDA(cudaEventRecord(stop_event, stream));
            CHECK_CUDA(cudaDeviceSynchronize());
            
            unsigned long long energy_after = getGPUEnergy(nvml_device);
            auto wall_end = std::chrono::steady_clock::now();
            
            // Calculate times
            float gpu_e2e_ms, kernel_ms;
            CHECK_CUDA(cudaEventElapsedTime(&gpu_e2e_ms, start_event, stop_event));
            CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
            float gpu_e2e_s = gpu_e2e_ms / 1000.0f;
            float kernel_s = kernel_ms / 1000.0f;
            float wall_s = std::chrono::duration<float>(wall_end - wall_start).count();
            
            double energy_j = (energy_after > energy_before) ? ((energy_after - energy_before) / 1000.0) : 0.0;
            double avg_power = (wall_s > 0) ? (energy_j / wall_s) : 0.0;
            bool below_target = (gpu_e2e_s < TARGET_RUNTIME_S);
            
            GPUTelemetry telem = getGPUTelemetry(nvml_device);
            
            writeCSVRow(csv_file, run_id_global, run_id_per_size, device_name, shape_id, batches,
                       gpu_e2e_s, kernel_s, wall_s, energy_j, avg_power, below_target,
                       shape.flops_per_batch, telem);
            csv_file.flush();
            
            // Progress
            double gflops = (kernel_s > 0) ? (shape.flops_per_batch * batches / kernel_s / 1e9) : 0.0;
            char check = below_target ? '!' : '+';
            std::cout << "  " << check << " [" << (rep + 1) << "/" << MACRO_REPEATS << "] "
                      << std::fixed << std::setprecision(3) << kernel_s << "s kernel, "
                      << gpu_e2e_s << "s e2e | "
                      << std::setprecision(1) << gflops << " GFLOPS, "
                      << energy_j << "J, " << (int)avg_power << "W, " << telem.temp << "°C\n";
            
            run_id_global++;
            run_id_per_size++;
            total_rows++;
            
            if (test_mode && total_rows >= 5) {
                std::cout << "\nTest mode: 5 rows written!\n";
                goto cleanup;
            }
        }
        
        if (shape_id != shapes_to_test.back()) {
            std::cout << "  Cooling 30s...\n";
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
    }
    
cleanup:
    csv_file.close();
    std::cout << "\nBenchmark complete! Results: " << output_file << "\n";
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaEventDestroy(start_kernel));
    CHECK_CUDA(cudaEventDestroy(stop_kernel));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_checksum));
    CHECK_CUDA(cudaStreamDestroy(stream));
    free(h_input); free(h_filter); free(h_output);
    nvmlShutdown();
    
    return EXIT_SUCCESS;
}
