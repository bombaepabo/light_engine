#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for Q8 block-quantized matrix multiplication
// Grid size: (rows + 64 - 1) / 64 threads
__global__ void matmul_q8_kernel(float* out, const float* x, const int8_t* w_weights, const float* w_scales, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float total_sum = 0.0f;
        int blocks_per_row = cols / 32;
        
        // Loop through each block of 32 elements in the row
        for (int b = 0; b < blocks_per_row; ++b) {
            float scale = w_scales[row * blocks_per_row + b];
            float block_sum = 0.0f;
            int weight_offset = row * cols + b * 32;
            int x_offset = b * 32;
            
            // Unroll or compile-time unroll this loop of 32
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                block_sum += (float)w_weights[weight_offset + j] * x[x_offset + j];
            }
            total_sum += block_sum * scale;
        }
        out[row] = total_sum;
    }
}

// C FFI helper to upload data to GPU VRAM
extern "C" void* gpu_upload(const void* host_ptr, size_t size) {
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for size %zu: %s\n", size, cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_ptr);
        return nullptr;
    }
    return device_ptr;
}

// C FFI helper to free GPU VRAM
extern "C" void gpu_free(void* device_ptr) {
    if (device_ptr != nullptr) {
        cudaFree(device_ptr);
    }
}

// C FFI helper to allocate clean uninitialized VRAM
extern "C" void* gpu_alloc(size_t size) {
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for size %zu: %s\n", size, cudaGetErrorString(err));
        return nullptr;
    }
    return device_ptr;
}

// C FFI helper to copy data back to CPU
extern "C" void gpu_copy_to_host(void* host_dest, const void* device_src, size_t size) {
    cudaMemcpy(host_dest, device_src, size, cudaMemcpyDeviceToHost);
}

// C FFI helper to copy data to GPU
extern "C" void gpu_copy_to_device(void* device_dest, const void* host_src, size_t size) {
    cudaMemcpy(device_dest, host_src, size, cudaMemcpyHostToDevice);
}

// C FFI helper to launch the kernel using pre-allocated/uploaded pointers
extern "C" void matmul_q8_gpu_launch(
    float* out_device,
    const float* x_device,
    const int8_t* w_weights_device,
    const float* w_scales_device,
    int rows,
    int cols
) {
    int threadsPerBlock = 64;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    
    matmul_q8_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        out_device,
        x_device,
        w_weights_device,
        w_scales_device,
        rows,
        cols
    );
    
    // Synchronize to make sure there are no race conditions
    cudaDeviceSynchronize();
}