extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/worker/include/blake2b.c")
        .include("src/worker/include")
        .compile("libblake2b.a");

    cc::Build::new()
        .file("src/worker/include/cuckoo-avx512.c")
        .include("src/worker/include")
        .flag("/arch:AVX512")
        .compile("libcuckoo.a.1");

    cc::Build::new()
        .file("src/worker/include/cuckoo.c")
        .include("src/worker/include")
        .flag("/arch:AVX2")
        .compile("libcuckoo.a.2");

    // Add link directory
    // - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
    // - This should be set by `$LIBRARY_PATH`
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64");
    #[cfg(feature = "gpu")]
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=miner\\src\\worker\\include");
}
