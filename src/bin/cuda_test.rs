// Declare the external C++ CUDA function
unsafe extern "C" {
    fn launch_matmul_gpu(out: *mut f32, x: *const f32, w: *const f32, rows: i32, cols: i32);
}

fn main() {
    println!("Initializing CUDA test...");

    // Define test Matrix (2x3) and Vector (3 elements)
    let w_data = [
        1.0f32, 2.0, 3.0, // Row 0
        4.0, 5.0, 6.0,    // Row 1
    ];
    let x_data = [1.0f32, 1.0, 1.0];
    let mut out_data = [0.0f32; 2];

    println!("Running matrix multiplication on NVIDIA GPU...");
    unsafe {
        // Call the GPU kernel wrapper
        launch_matmul_gpu(
            out_data.as_mut_ptr(),
            x_data.as_ptr(),
            w_data.as_ptr(),
            2, // rows
            3, // cols
        );
    }

    println!("Result computed on GPU:");
    println!("Row 0: {} (Expected: 6.0)", out_data[0]);
    println!("Row 1: {} (Expected: 15.0)", out_data[1]);
}