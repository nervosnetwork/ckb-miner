extern crate cc;

fn main() {
    cc::Build::new()
        .file("src/worker/blake2b.c")
        .include("src/worker")
        .flag("-O3")
        .flag("-msse4")
        .static_flag(true)
        .compile("libblake2b.a");

    cc::Build::new()
        .file("src/worker/cuckoo-avx512.c")
        .include("src/worker")
        .flag("-O3")
        .flag("-lcrypto")
        .flag("-mavx512f")
        .flag("-mavx512cd")
        .flag("-msse4")
        .static_flag(true)
        .compile("libcuckoo.a.1");

    cc::Build::new()
        .file("src/worker/cuckoo.c")
        .include("src/worker")
        .flag("-O3")
        .flag("-lcrypto")
        .flag("-mavx2")
        .flag("-msse4")
        .static_flag(true)
        .compile("libcuckoo.a.2");

    cc::Build::new()
        .file("src/worker/cuckoo.cu")
        .include("src/worker")
        .flag("-O3")
        .flag("-lcrypto")
        .cuda(true)
        .compile("libcuckoo.a.3");

    // Add link directory
    // - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
    // - This should be set by `$LIBRARY_PATH`
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
