use std::path::Path;
use std::env;

fn main() {
    if cfg!(windows) {
        cc::Build::new()
            .file("src/worker/include/eaglesong.c")
            .include("src/worker/include")
            .compile("libeaglesong.a.0");

        cc::Build::new()
            .file("src/worker/include/eaglesong_avx2.c")
            .include("src/worker/include")
            .flag("/arch:AVX2")
            .compile("libeaglesong.a.1");

        cc::Build::new()
            .file("src/worker/include/eaglesong_avx512.c")
            .include("src/worker/include")
            .flag("/arch:AVX512")
            .compile("libeaglesong.a.2");

        #[cfg(feature = "gpu")]
        cc::Build::new()
            .file("src/worker/include/eaglesong.cu")
            .include("src/worker/include")
            .flag("-O3")
            .cuda(true)
            .compile("libeaglesong.a.3");

        #[cfg(feature = "gpu")]
        {
            if let Some(lib_dir) = env::var_os("CUDA_LIB_DIR") {
                let lib_dir = Path::new(&lib_dir);
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
            } else {
                // - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
                // - default cuda lib path for CUDA 10.1
                println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64");
            }
            println!("cargo:rustc-link-lib=cudart");
        }
    } else {
        cc::Build::new()
            .file("src/worker/include/eaglesong.c")
            .include("src/worker/include")
            .flag("-O3")
            .flag("-lcrypto")
            .static_flag(true)
            .compile("libeaglesong.a.0");

        cc::Build::new()
            .file("src/worker/include/eaglesong_avx2.c")
            .include("src/worker/include")
            .flag("-O3")
            .flag("-lcrypto")
            .flag("-mavx2")
            .static_flag(true)
            .compile("libeaglesong.a.1");

        cc::Build::new()
            .file("src/worker/include/eaglesong_avx512.c")
            .include("src/worker/include")
            .flag("-O3")
            .flag("-lcrypto")
            .flag("-mavx512f")
            .static_flag(true)
            .compile("libeaglesong.a.2");

        #[cfg(feature = "gpu")]
        cc::Build::new()
            .file("src/worker/include/eaglesong.cu")
            .include("src/worker/include")
            .flag("-O3")
            .flag("-lcrypto")
            .cuda(true)
            .compile("libeaglesong.a.3");

        #[cfg(feature = "gpu")]
        {
            if let Some(lib_dir) = env::var_os("CUDA_LIB_DIR") {
                let lib_dir = Path::new(&lib_dir);
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
            } else {
                // - This path depends on where you install CUDA (i.e. depends on your Linux distribution)
                // - default cuda lib path for Ubuntu 18.04
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            }
            println!("cargo:rustc-link-lib=cudart");
        }        
    }
}
