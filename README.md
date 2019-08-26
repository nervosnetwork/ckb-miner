# ckb-miner

| PoW algorithm | Since | 
| ----          | ----    |
|   Eaglesong   | v0.19.0 |

A miner for avx2 CPUs, avx512 CPUs and Nvidia GPUs.

Using avx512, the hash rate can be up to 5M per thread on 2.5Ghz CPUs.

For GPUs, the hash rate is 190M on GTX 1060, 300M on GTX 1070ti.

## How to build
By default, GPU miner is not built, you can build GPU miner with this parameters:

For Linux and OSX:

```
export CUDA_LIB_DIR=<PATH> # <PATH> is the path where you install CUDA, eg export CUDA_LIB_DIR=/usr/local/cuda/lib64
cargo build --release --features gpu
```

For windows:

```
set CUDA_LIB_DIR=<PATH> # <PATH> is the path where you install CUDA, eg set CUDA_LIB_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64
cargo build --release --features gpu
```

Cuda is essential to build and run GPU miners.

## How to use
1. Modify `ckb-miner.toml` in this repo and copy it to your working directory. You can specify the CPUs and GPUs you want to use.

2. Copy `ckb-miner` (you can build it yourself or download it) to your working directory.

3. run the miner

```
./ckb-miner
```

