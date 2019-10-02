use super::{Worker, WorkerMessage};
use ckb_logger::{debug, error};
use ckb_types::{packed::Byte32, prelude::*, U256};
use crossbeam_channel::{Receiver, Sender};
use indicatif::ProgressBar;
use std::thread;
use std::time::{Duration, Instant};

const STATE_UPDATE_DURATION_MILLIS: u128 = 300;

extern "C" {
    pub fn c_solve_cl(
        input: *const u8,
        target: *const u8,
        nonce: *mut u64,
        plat_id: u32,
        gpuid: u32,
    ) -> u32;
    pub fn c_plat_init(plat_id: u32) -> u32;
}

pub fn plat_init(plat_id: u32) -> u32 {
    unsafe { c_plat_init(plat_id) }
}

pub struct EaglesongCL {
    start: bool,
    pow_info: Option<(Byte32, U256)>,
    seal_tx: Sender<(Byte32, u64)>,
    worker_rx: Receiver<WorkerMessage>,
    seal_candidates_found: u64,
    plat_id: u32,
    gpuid: u32,
}

impl EaglesongCL {
    pub fn new(
        seal_tx: Sender<(Byte32, u64)>,
        worker_rx: Receiver<WorkerMessage>,
        plat_id: u32,
        gpuid: u32,
    ) -> Self {
        Self {
            start: true,
            pow_info: None,
            seal_candidates_found: 0,
            seal_tx,
            worker_rx,
            plat_id,
            gpuid,
        }
    }

    fn poll_worker_message(&mut self) {
        if let Ok(msg) = self.worker_rx.try_recv() {
            match msg {
                WorkerMessage::NewWork(pow_info) => {
                    self.pow_info = Some(pow_info);
                }
                WorkerMessage::Stop => {
                    self.start = false;
                }
                WorkerMessage::Start => {
                    self.start = true;
                }
            }
        }
    }

    #[inline]
    fn solve(&mut self, pow_hash: &Byte32, target: &U256) -> usize {
        unsafe {
            let mut nonce = 0u64;
            let ns = c_solve_cl(
                pow_hash.as_slice().as_ptr(),
                target.to_be_bytes().as_ptr(),
                &mut nonce,
                self.plat_id,
                self.gpuid,
            );
            if nonce != 0 {
                debug!(
                    "send new found seal, pow_hash {:x}, nonce {:?}",
                    pow_hash, nonce
                );
                if let Err(err) = self.seal_tx.send((pow_hash.clone(), nonce)) {
                    error!("seal_tx send error {:?}", err);
                }
                self.seal_candidates_found += 1;
            }

            return ns as usize;
        }
    }
}

impl Worker for EaglesongCL {
    fn run(&mut self, progress_bar: ProgressBar) {
        let mut state_update_counter = 0usize;
        let mut start = Instant::now();
        loop {
            self.poll_worker_message();
            if self.start {
                if let Some((pow_hash, target)) = self.pow_info.clone() {
                    state_update_counter += self.solve(&pow_hash, &target);

                    let elapsed = start.elapsed();
                    if elapsed.as_millis() > STATE_UPDATE_DURATION_MILLIS {
                        let elapsed_nanos: f64 = (elapsed.as_secs() * 1_000_000_000
                            + u64::from(elapsed.subsec_nanos()))
                            as f64
                            / 1_000_000_000.0;
                        progress_bar.set_message(&format!(
                            "hash rate: {:>10.3} / seals found: {:>10}",
                            state_update_counter as f64 / elapsed_nanos,
                            self.seal_candidates_found,
                        ));
                        progress_bar.inc(1);
                        state_update_counter = 0;
                        start = Instant::now();
                    }
                }
            } else {
                // reset state and sleep
                state_update_counter = 0;
                start = Instant::now();
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
