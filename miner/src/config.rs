use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinerConfig {
    pub rpc_url: String,
    pub poll_interval: u64,
    pub block_on_submit: bool,
    pub cpus: u32,
    pub gpus: Vec<GpuConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuConfig {
    pub plat_id: u32,
    pub gpu_ids: Vec<u32>,
}
