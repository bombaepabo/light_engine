#include <cuda_runtime.h>

// This is the GPU kernel. Each thread computes one row of the output.
__global__ void matmul_kernel(float* out, const float* x, const float* w, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += w[row * cols + col] * x[col];
        }
        out[row] = sum;
    }
}

// C-compatible wrapper that Rust can call to launch the GPU kernel
extern "C" void launch_matmul_gpu(float* out, const float* x, const float* w, int rows, int cols) {
    float *d_out, *d_x, *d_w;
    
    // Allocate memory on the GPU (VRAM)
    cudaMalloc(&d_out, rows * sizeof(float));
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_w, rows * cols * sizeof(float));
    
    // Copy data from Host (CPU RAM) to Device (GPU VRAM)
    cudaMemcpy(d_x, x, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the kernel with enough GPU threads
    int threadsPerBlock = 64;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_x, d_w, rows, cols);
    
    // Copy the result back from GPU VRAM to CPU RAM
    cudaMemcpy(out, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_out);
    cudaFree(d_x);
    cudaFree(d_w);
}