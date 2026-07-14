fn main() {
    // Tell Cargo to compile kernel.cu using nvcc (CUDA compiler)
    cc::Build::new()
        .cuda(true)
        .file("kernel.cu")
        .compile("cuda_kernels");

    // Link against CUDA runtime libraries
    println!("cargo:rustc-link-lib=dylib=cudart");
}