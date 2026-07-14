// Declare the external C++ CUDA functions
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

fn main() {
    println!("Initializing CUDA Q8 test...");
    
    // We will test 1 row with 32 columns
    let x = [1.0f32; 32];
    let w_weights = [2i8; 32];   // 32 weights set to 2
    let w_scales = [0.5f32; 1];  // 1 scale block set to 0.5
    let mut out = [0.0f32; 1];   // output vector of size 1

    // Expected output: sum_i(x_i * weight_i) * scale = (32 * 1.0 * 2) * 0.5 = 32.0

    unsafe {
        // Allocate VRAM
        let x_device = gpu_alloc(32 * 4) as *mut f32;
        let w_weights_device = gpu_alloc(32) as *mut i8;
        let w_scales_device = gpu_alloc(4) as *mut f32;
        let out_device = gpu_alloc(4) as *mut f32;

        // Copy host memory to device VRAM
        gpu_copy_to_device(x_device as *mut _, x.as_ptr() as *const _, 32 * 4);
        gpu_copy_to_device(w_weights_device as *mut _, w_weights.as_ptr() as *const _, 32);
        gpu_copy_to_device(w_scales_device as *mut _, w_scales.as_ptr() as *const _, 4);

        // Run the Q8 matmul GPU kernel
        matmul_q8_gpu_launch(
            out_device,
            x_device,
            w_weights_device,
            w_scales_device,
            1,  // 1 row
            32, // 32 columns
        );

        // Copy results back to host memory
        gpu_copy_to_host(out.as_mut_ptr() as *mut _, out_device as *const _, 4);

        // Free VRAM
        gpu_free(x_device as *mut _);
        gpu_free(w_weights_device as *mut _);
        gpu_free(w_scales_device as *mut _);
        gpu_free(out_device as *mut _);
    }

    println!("GPU Q8 Matmul Result: {} (Expected: 32.0)", out[0]);
    
    if (out[0] - 32.0).abs() < 1e-5 {
        println!("Test Passed successfully!");
    } else {
        println!("Test Failed!");
    }
}