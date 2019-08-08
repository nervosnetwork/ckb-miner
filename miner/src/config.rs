use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinerConfig {
    pub rpc_url: String,
    pub poll_interval: u64,
    pub block_on_submit: bool,
    pub cpus: Vec<CpuConfig>,
    pub gpus: GpuConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuConfig {
    pub gpuids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Copy)]
pub struct CpuConfig {
    pub arch: u32,
    pub threads: u32,
}
