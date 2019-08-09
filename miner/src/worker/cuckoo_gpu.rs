use super::{Worker, WorkerMessage, CYCLE_LEN};
use byteorder::{ByteOrder, LittleEndian};
use ckb_core::header::Seal;
use ckb_logger::{debug, error};
use crossbeam_channel::{Receiver, Sender};
use indicatif::ProgressBar;
use numext_fixed_hash::H256;
use std::thread;
use std::time::{Duration, Instant};

const STATE_UPDATE_DURATION_MILLIS: u128 = 300;

#[link(name="cuckoo", kind="static")]
extern "C" {
    pub fn c_solve_gpu(
        output: *mut u32,
        nonce: *mut u64,
        input: *const u8,
        target: *const u8,
        gpuid: u32,
    ) -> u32;
}

pub struct CuckooGpu {
    start: bool,
    pow_info: Option<(H256, H256)>,
    seal_tx: Sender<(H256, Seal)>,
    worker_rx: Receiver<WorkerMessage>,
    seal_candidates_found: u64,
    gpuid: u32,
}

impl CuckooGpu {
    pub fn new(
        seal_tx: Sender<(H256, Seal)>,
        worker_rx: Receiver<WorkerMessage>,
        gpuid: u32,
    ) -> Self {
        Self {
            start: true,
            pow_info: None,
            seal_candidates_found: 0,
            seal_tx,
            worker_rx,
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
    fn solve(&mut self, pow_hash: &H256, target: &H256) -> usize {
        unsafe {
            let mut nonce = 0u64;
            let mut output = vec![0u32; CYCLE_LEN + 1];
            let ns = c_solve_gpu(
                output.as_mut_ptr(),
                &mut nonce,
                pow_hash[..].as_ptr(),
                target[..].as_ptr(),
                self.gpuid,
            );
            if ns > 0 && output[CYCLE_LEN] == 1 {
                let mut proof_u8 = vec![0u8; CYCLE_LEN << 2];
                LittleEndian::write_u32_into(&output[0..CYCLE_LEN], &mut proof_u8);
                let seal = Seal::new(nonce, proof_u8.into());
                debug!(
                    "send new found seal, pow_hash {:x}, seal {:?}",
                    pow_hash, seal
                );
                if let Err(err) = self.seal_tx.send((pow_hash.clone(), seal)) {
                    error!("seal_tx send error {:?}", err);
                }
                self.seal_candidates_found += 1;
            }

            return ns as usize;
        }
    }
}

impl Worker for CuckooGpu {
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
                            "gps: {:>10.3} / cycles found: {:>10}",
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
