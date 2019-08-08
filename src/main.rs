mod exit_code;
mod sentry;

use crate::exit_code::ExitCode;
use crate::sentry::{sentry_init, SentryConfig};
use ckb_build_info::Version;
use ckb_logger::Config as LogConfig;
use miner::{Client, Miner, MinerConfig};
use serde_derive::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

pub const MINER_CONFIG_FILE_NAME: &str = "ckb-miner.toml";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub logger: LogConfig,
    pub sentry: SentryConfig,
    pub miner: MinerConfig,
}

fn main() {
    if let Ok(config) = read_config(None) {
        let version = get_version();
        let _logger_guard = ckb_logger::init(config.logger).expect("Init logger failed!");
        let _sentry_guard = sentry_init(&config.sentry, &version);

        let client = Client::new(config.miner.clone());
        let mut miner = Miner::new(client, config.miner);

        miner.run();
    }
}

fn read_config(cfg_path: Option<String>) -> Result<AppConfig, ExitCode> {
    let cfg_path = match cfg_path {
        Some(s) => PathBuf::from(s),
        None => ::std::env::current_dir()?.join(MINER_CONFIG_FILE_NAME),
    };
    let data = fs::read(cfg_path)?;
    let config = toml::from_slice(&data)?;
    Ok(config)
}

fn get_version() -> Version {
    let major = env!("CARGO_PKG_VERSION_MAJOR")
        .parse::<u8>()
        .expect("CARGO_PKG_VERSION_MAJOR parse success");
    let minor = env!("CARGO_PKG_VERSION_MINOR")
        .parse::<u8>()
        .expect("CARGO_PKG_VERSION_MINOR parse success");
    let patch = env!("CARGO_PKG_VERSION_PATCH")
        .parse::<u16>()
        .expect("CARGO_PKG_VERSION_PATCH parse success");
    let dash_pre = {
        let pre = env!("CARGO_PKG_VERSION_PRE");
        if pre == "" {
            pre.to_string()
        } else {
            "-".to_string() + pre
        }
    };

    let commit_describe = option_env!("COMMIT_DESCRIBE").map(ToString::to_string);
    #[cfg(docker)]
    let commit_describe = commit_describe.map(|s| s.replace("-dirty", ""));
    let commit_date = option_env!("COMMIT_DATE").map(ToString::to_string);
    let code_name = Some("rylai-v8".to_string());
    Version {
        major,
        minor,
        patch,
        dash_pre,
        code_name,
        commit_describe,
        commit_date,
    }
}
