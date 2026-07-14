use rayon::prelude::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use crate::model::QuantizedTensor;

unsafe extern "C" {
    fn gpu_alloc(size: usize) -> *mut std::ffi::c_void;
    fn gpu_free(device_ptr: *mut std::ffi::c_void);
    fn gpu_copy_to_device(device_dest: *mut std::ffi::c_void, host_src: *const std::ffi::c_void, size: usize);
    fn gpu_copy_to_host(host_dest: *mut std::ffi::c_void, device_src: *const std::ffi::c_void, size: usize);
    fn matmul_q8_gpu_launch(
        out_device: *mut f32,
        x_device: *const f32,
        w_weights_device: *const i8,
        w_scales_device: *const f32,
        rows: i32,
        cols: i32,
    );
}

// 1. Root Mean Square Normalization (RMSNorm)
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], epsilon: f32) {
    // Calculate the sum of squares: x_0^2 + x_1^2 + ...
    let mut ss = x.iter().map(|&val| val * val).sum::<f32>();
    
    // Calculate the mean (average) of the squares
    ss /= x.len() as f32;
    ss += epsilon; // Add tiny constant to avoid division by zero
    
    // Scale factor is 1 / sqrt(mean_squares)
    let scale = 1.0 / ss.sqrt();
    
    // Normalize each element and scale it by the learned weight parameter
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

// 2. Stabilized Softmax
pub fn softmax(x: &mut [f32]) {
    // Find the maximum value in the vector for numerical stability (prevents float overflow)
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Subtract max_val, compute the exponent (e^x), and sum them up
    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    
    // Divide each exponent by the sum so all probabilities add up to 1.0
    for val in x.iter_mut() {
        *val /= sum;
    }
}

// 3. Matrix-Vector Multiplication (Scalar Fallback)
pub fn matmul_scalar(out: &mut [f32], x: &[f32], w: &[f32], _rows: usize, cols: usize) {
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_offset = i * cols;
        let mut sum = 0.0;
        for j in 0..cols {
            sum += w[row_offset + j] * x[j];
        }
        *val = sum;
    });
}
// 3b. AVX2 SIMD Matrix-Vector Multiplication
// - target_feature(enable = "avx2") instructs the compiler to generate specialized AVX2 CPU instructions.
// - This function is unsafe because calling it on a CPU that doesn't support AVX2 will crash the program.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_avx2(out: &mut [f32], x: &[f32], w: &[f32], _rows: usize, cols: usize) {
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_offset = i * cols;
        
        // Wrap the body in unsafe { ... } to keep the Rust 2024 compiler happy:
        unsafe {
            let mut sum_vec = _mm256_setzero_ps(); // Vector accumulator: [0.0; 8]
            
            let mut j = 0;
            // Process columns in chunks of 8
            while j + 8 <= cols {
                let w_ptr = w.as_ptr().add(row_offset + j);
                let x_ptr = x.as_ptr().add(j);
                
                // 1. Load 8 weights and 8 activations
                let w_val = _mm256_loadu_ps(w_ptr);
                let x_val = _mm256_loadu_ps(x_ptr);
                
                // 2. Fused Multiply-Add: sum_vec = (w_val * x_val) + sum_vec
                sum_vec = _mm256_fmadd_ps(w_val, x_val, sum_vec);
                j += 8;
            }
            
            // 3. Write vector to a temporary float array to do horizontal sum
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), sum_vec);
            let mut sum = tmp.iter().sum::<f32>();
            
            // 4. Process any leftover elements if cols is not a multiple of 8
            while j < cols {
                sum += w[row_offset + j] * x[j];
                j += 1;
            }
            
            *val = sum;
        }
    });
}
// 3c. Matmul wrapper with runtime CPU feature detection
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], rows: usize, cols: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: we checked that the CPU supports AVX2
            unsafe {
                matmul_avx2(out, x, w, rows, cols);
            }
            return;
        }
    }
    // Fall back to scalar multi-threaded loop if AVX2 is not supported
    matmul_scalar(out, x, w, rows, cols);
}

// 3d. Quantized Matrix-Vector Multiplication (Scalar Fallback)
pub fn matmul_q8_scalar(out: &mut [f32], x: &[f32], w: &QuantizedTensor, _rows: usize, cols: usize) {
    let block_size = 32;
    let blocks_per_row = cols / block_size;

    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_offset = i * cols;
        let scale_offset = i * blocks_per_row;
        let mut total_sum = 0.0;

        for b in 0..blocks_per_row {
            let scale = w.scales[scale_offset + b];
            let block_weight_offset = row_offset + b * block_size;
            let block_x_offset = b * block_size;
            
            let mut block_sum = 0.0;
            for j in 0..block_size {
                block_sum += w.weights[block_weight_offset + j] as f32 * x[block_x_offset + j];
            }
            total_sum += block_sum * scale;
        }
        *val = total_sum;
    });
}

// 3e. Quantized Matrix-Vector Multiplication (AVX2 Optimized)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_q8_avx2(out: &mut [f32], x: &[f32], w: &QuantizedTensor, _rows: usize, cols: usize) {
    let block_size = 32;
    let blocks_per_row = cols / block_size;

    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let row_offset = i * cols;
        let scale_offset = i * blocks_per_row;
        let mut total_sum = 0.0;

        unsafe {
            for b in 0..blocks_per_row {
                let scale = w.scales[scale_offset + b];
                let block_weight_offset = row_offset + b * block_size;
                let block_x_offset = b * block_size;

                let mut sum_vec = _mm256_setzero_ps();

                // Process the block of size 32 in 4 chunks of 8
                for k in (0..block_size).step_by(8) {
                    let w_ptr = w.weights.as_ptr().add(block_weight_offset + k);
                    // Load 8 i8 elements (64 bits) and sign-extend to i32, then convert to f32
                    let w_128 = _mm_loadl_epi64(w_ptr as *const __m128i);
                    let w_i32 = _mm256_cvtepi8_epi32(w_128);
                    let w_f32 = _mm256_cvtepi32_ps(w_i32);

                    // Load 8 float elements of x
                    let x_ptr = x.as_ptr().add(block_x_offset + k);
                    let x_val = _mm256_loadu_ps(x_ptr);

                    sum_vec = _mm256_fmadd_ps(w_f32, x_val, sum_vec);
                }

                // Horizontal sum of the 8 float accumulators
                let mut tmp = [0.0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), sum_vec);
                let block_sum = tmp.iter().sum::<f32>();

                total_sum += block_sum * scale;
            }
        }
        *val = total_sum;
    });
}

// 3f. Quantized Matrix-Vector Multiplication on GPU using CUDA
pub fn matmul_q8_gpu(out: &mut [f32], x: &[f32], w: &QuantizedTensor, rows: usize, cols: usize) {
    let x_size = x.len() * std::mem::size_of::<f32>();
    let out_size = out.len() * std::mem::size_of::<f32>();
    
    unsafe {
        // Allocate temp GPU buffers for input x and output
        let x_device = gpu_alloc(x_size) as *mut f32;
        let out_device = gpu_alloc(out_size) as *mut f32;
        
        // Copy x from CPU to GPU
        gpu_copy_to_device(x_device as *mut _, x.as_ptr() as *const _, x_size);
        
        // Launch the GPU kernel
        matmul_q8_gpu_launch(
            out_device,
            x_device,
            w.weights_device,
            w.scales_device,
            rows as i32,
            cols as i32,
        );
        
        // Copy output back from GPU to CPU
        gpu_copy_to_host(out.as_mut_ptr() as *mut _, out_device as *const _, out_size);
        
        // Free temp GPU buffers
        gpu_free(x_device as *mut _);
        gpu_free(out_device as *mut _);
    }
}

// 3g. Quantized Matrix-Vector Multiplication Wrapper
pub fn matmul_q8(out: &mut [f32], x: &[f32], w: &QuantizedTensor, rows: usize, cols: usize) {
    matmul_q8_gpu(out, x, w, rows, cols);
}

// 4. Rotary Position Embedding (RoPE)
// Rotates the Query or Key vectors to inject positional information.
// - vec: the query or key vector for the current token (size: n_heads * head_size)
// - pos: the position of the token in the sentence (0, 1, 2...)
// - freq_cis_real & freq_cis_imag: the precompiled rotation tables loaded in Phase 1
// - n_heads: number of attention heads (6 in our model)
// - head_size: size of each head (288 / 6 = 48 in our model)
pub fn rope(
    vec: &mut [f32],
    pos: usize,
    freq_cis_real: &[f32],
    freq_cis_imag: &[f32],
    n_heads: usize,
    head_size: usize,
) {
    // Loop through each attention head
    for h in 0..n_heads {
        let head_offset = h * head_size;
        
        // Loop through elements of this head in pairs (2i, 2i+1)
        for i in (0..head_size).step_by(2) {
            let v0 = vec[head_offset + i];
            let v1 = vec[head_offset + i + 1];
            
            // Find the index in our precompiled sine/cosine tables
            // Offset logic: each position `pos` has `head_size / 2` rotation frequencies.
            let freq_idx = pos * (head_size / 2) + (i / 2);
            let f_real = freq_cis_real[freq_idx];
            let f_imag = freq_cis_imag[freq_idx];
            
            // Perform 2D complex multiplication (rotation)
            vec[head_offset + i] = v0 * f_real - v1 * f_imag;
            vec[head_offset + i + 1] = v0 * f_imag + v1 * f_real;
        }
    }
}
pub fn silu_and_mul(hb: &mut [f32], hb2: &[f32]) {
    for i in 0..hb.len() {
        let x = hb[i];
        // Sigmoid formula: 1.0 / (1.0 + e^-x)
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        hb[i] = x * sigmoid * hb2[i];
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rmsnorm() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0; 4];
        
        // Run RMSNorm with epsilon = 1e-5
        rmsnorm(&mut out, &x, &weight, 1e-5);
        
        // The RMS value of [1, 2, 3, 4] is sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386
        // So the outputs should be [x_i / 2.7386]
        let expected = [
            1.0 / 2.7386128,
            2.0 / 2.7386128,
            3.0 / 2.7386128,
            4.0 / 2.7386128,
        ];
        
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-5);
        }
    }
    #[test]
    fn test_softmax() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        
        // Sum of exps: e^1 + e^2 + e^3 = 2.718 + 7.389 + 20.085 = 30.192
        // Expected: [e^1 / sum, e^2 / sum, e^3 / sum]
        let expected = [0.09003057, 0.24472847, 0.66524096];
        
        for i in 0..3 {
            assert!((x[i] - expected[i]).abs() < 1e-5);
        }
    }
    #[test]
    fn test_matmul() {
        // A 2x3 matrix (2 rows, 3 columns)
        let w = [
            1.0, 2.0, 3.0,  // Row 0
            4.0, 5.0, 6.0,  // Row 1
        ];
        let x = [1.0, 1.0, 1.0]; // Input vector
        let mut out = [0.0; 2];  // Output vector (size = rows = 2)
        
        matmul(&mut out, &x, &w, 2, 3);
        
        // Row 0: 1*1 + 2*1 + 3*1 = 6.0
        // Row 1: 4*1 + 5*1 + 6*1 = 15.0
        assert_eq!(out[0], 6.0);
        assert_eq!(out[1], 15.0);
    }
      #[test]
    fn test_rope() {
        let mut vec = [1.0, 2.0, 3.0, 4.0];
        // Precompiled cos and sin values for index 0
        let freq_real = [0.8, 0.6];
        let freq_imag = [0.5, 0.8];
        
        // We rotate a vector with 2 heads, head_size = 2, at pos = 0
        rope(&mut vec, 0, &freq_real, &freq_imag, 2, 2);
        
        // Expected calculations (both heads use index 0 frequencies):
        // Pair 1: (1, 2) rotated by (0.8, 0.5) -> (-0.2, 2.1)
        // Pair 2: (3, 4) rotated by (0.8, 0.5) -> (0.4, 4.7)
        assert!((vec[0] - (-0.2)).abs() < 1e-5);
        assert!((vec[1] - 2.1).abs() < 1e-5);
        assert!((vec[2] - 0.4).abs() < 1e-5);
        assert!((vec[3] - 4.7).abs() < 1e-5);
    }

      #[test]
    fn test_matmul_q8() {
        // Mock a quantized tensor of 2 rows, cols = 32
        let weights = vec![1i8; 32].into_iter().chain(vec![2i8; 32].into_iter()).collect::<Vec<i8>>();
        let scales = vec![1.5f32, 2.0f32];
        let w = QuantizedTensor {
            scales: &scales,
            weights: &weights,
        };
        let x = vec![1.0f32; 32];
        let mut out = [0.0; 2];
        
        matmul_q8(&mut out, &x, &w, 2, 32);
        
        // Expected calculations:
        // Row 0: (32 elements * 1 weight * 1.0 input) * 1.5 scale = 48.0
        // Row 1: (32 elements * 2 weight * 1.0 input) * 2.0 scale = 128.0
        assert!((out[0] - 48.0).abs() < 1e-5);
        assert!((out[1] - 128.0).abs() < 1e-5);
    }
}