# ckb-miner

``Current PoW algorithm: Eaglesong.``

A miner for avx2 CPUs, avx512 CPUs and Nvidia GPUs.

Using avx512, the hash rate can be up to 5M per thread on 2.5Ghz CPUs.

For GPUs, the hash rate is 190M on GTX 1060, 300M on GTX 1070ti.

## How to build
By default, GPU miner is not built, you can build GPU miner with this parameters:
```
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

