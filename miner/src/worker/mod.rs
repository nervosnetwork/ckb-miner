mod cuckoo;
#[cfg(feature = "gpu")]
mod cuckoo_gpu;

use crate::MinerConfig;
use ckb_core::header::Seal;
use ckb_logger::error;
use crossbeam_channel::{unbounded, Sender};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use numext_fixed_hash::H256;
use std::thread;

const CYCLE_LEN: usize = 12;

#[derive(Clone)]
pub enum WorkerMessage {
    Stop,
    Start,
    NewWork((H256, H256)),
}

#[derive(Clone)]
pub struct WorkerController {
    inner: Vec<Sender<WorkerMessage>>,
}

impl WorkerController {
    pub fn new(inner: Vec<Sender<WorkerMessage>>) -> Self {
        Self { inner }
    }

    pub fn send_message(&self, message: WorkerMessage) {
        for worker_tx in self.inner.iter() {
            if let Err(err) = worker_tx.send(message.clone()) {
                error!("worker_tx send error {:?}", err);
            };
        }
    }
}

const PROGRESS_BAR_TEMPLATE: &str = "{prefix:.bold.dim} {spinner:.green} [{elapsed_precise}] {msg}";

pub fn start_worker(
    config: MinerConfig,
    seal_tx: Sender<(H256, Seal)>,
    mp: &MultiProgress,
) -> WorkerController {
    let mut worker_txs = Vec::new();
    #[cfg(feature = "gpu")]
    for i in config.gpus {
        let worker_name = format!("Cuckoo-Worker-CPU-{}", i);
        // `100` is the len of progress bar, we can use any dummy value here,
        // since we only show the spinner in console.
        let pb = mp.add(ProgressBar::new(100));
        pb.set_style(ProgressStyle::default_bar().template(PROGRESS_BAR_TEMPLATE));
        pb.set_prefix(&worker_name);

        let (worker_tx, worker_rx) = unbounded();
        let seal_tx = seal_tx.clone();
        thread::Builder::new()
            .name(worker_name)
            .spawn(move || {
                let mut worker = cuckoo_gpu::CuckooGpu::new(seal_tx, worker_rx, i);
                worker.run(pb);
            })
            .expect("Start `CuckooGpu` worker thread failed");
        worker_txs.push(worker_tx);
    }

    let arch = if is_x86_feature_detected!("avx512f") { 1 } else { 0 };
    for i in 0..config.cpus {
        let worker_name = format!("Cuckoo-Worker-CPU-{}", i);
        // `100` is the len of progress bar, we can use any dummy value here,
        // since we only show the spinner in console.
        let pb = mp.add(ProgressBar::new(100));
        pb.set_style(ProgressStyle::default_bar().template(PROGRESS_BAR_TEMPLATE));
        pb.set_prefix(&worker_name);

        let (worker_tx, worker_rx) = unbounded();
        let seal_tx = seal_tx.clone();
        thread::Builder::new()
            .name(worker_name)
            .spawn(move || {
                let mut worker = cuckoo::CuckooCpu::new(seal_tx, worker_rx, arch);
                worker.run(pb);
            })
            .expect("Start `CuckooGpu` worker thread failed");
        worker_txs.push(worker_tx);
    }

    WorkerController::new(worker_txs)
}

pub trait Worker {
    fn run(&mut self, progress_bar: ProgressBar);
}
