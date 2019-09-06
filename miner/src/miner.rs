use crate::client::Client;
use crate::worker::{start_worker, WorkerController, WorkerMessage};
use crate::MinerConfig;
use crate::Work;
use ckb_logger::{debug, error, info};
use ckb_types::{packed::Byte32, packed::Header, prelude::*, utilities::difficulty_to_target};
use ckb_util::Mutex;
use crossbeam_channel::{unbounded, Receiver};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lru_cache::LruCache;
use std::sync::Arc;
use std::thread;
use std::time;

const WORK_CACHE_SIZE: usize = 32;

pub struct Miner {
    pub client: Client,
    pub works: Arc<Mutex<LruCache<Byte32, Work>>>,
    pub worker_controller: WorkerController,
    pub seal_rx: Receiver<(Byte32, u64)>,
    pub pb: ProgressBar,
    pub seals_found: u64,
    pub stderr_is_tty: bool,
}

impl Miner {
    pub fn new(client: Client, config: MinerConfig) -> Miner {
        let (seal_tx, seal_rx) = unbounded();
        let mp = MultiProgress::new();

        let worker_controller = start_worker(config.clone(), seal_tx.clone(), &mp);

        let pb = mp.add(ProgressBar::new(100));
        pb.set_style(ProgressStyle::default_bar().template("{msg:.green}"));

        let stderr_is_tty = console::Term::stderr().is_term();

        thread::spawn(move || {
            mp.join().expect("MultiProgress join failed");
        });

        let works = Arc::new(Mutex::new(LruCache::new(WORK_CACHE_SIZE)));

        {
            let (works, worker_controller, mut client) = (
                Arc::clone(&works),
                worker_controller.clone(),
                client.clone(),
            );

            thread::spawn(move || loop {
                if let Some(work) = client.poll_new_work() {
                    let pow_hash = work.block.header().calc_pow_hash();
                    let target =
                        difficulty_to_target(&work.block.header().raw().difficulty().unpack());
                    if works.lock().insert(pow_hash.clone(), work).is_none() {
                        worker_controller.send_message(WorkerMessage::NewWork((pow_hash, target)));
                    }
                }
                thread::sleep(time::Duration::from_millis(config.poll_interval));
            });
        }

        Miner {
            works: works,
            seals_found: 0,
            client,
            worker_controller,
            seal_rx,
            pb,
            stderr_is_tty,
        }
    }

    pub fn run(&mut self) {
        loop {
            match self.seal_rx.recv() {
                Ok((pow_hash, nonce)) => self.submit_seal(pow_hash, nonce),
                _ => {
                    error!("seal_rx closed");
                    break;
                }
            }
        }
    }

    fn submit_seal(&mut self, pow_hash: Byte32, nonce: u64) {
        let new_work = { self.works.lock().get_refresh(&pow_hash).cloned() };
        if let Some(work) = new_work {
            let raw_header = work.block.header().raw();
            let header = Header::new_builder()
                .raw(raw_header)
                .nonce(nonce.pack())
                .build();
            let block = work.block.as_builder().header(header).build().into_view();
            let block_hash: Byte32 = block.hash();
            if self.stderr_is_tty {
                debug!("Found! #{} {:#x}", block.number(), block_hash);
            } else {
                info!("Found! #{} {:#x}", block.number(), block_hash);
            }

            // submit block and poll new work
            {
                self.client
                    .submit_block(&work.work_id.to_string(), block.data());
                // poll new work
                if let Some(work) = self.client.poll_new_work() {
                    let pow_hash = work.block.header().calc_pow_hash();
                    let target =
                        difficulty_to_target(&work.block.header().raw().difficulty().unpack());
                    if self.works.lock().insert(pow_hash.clone(), work).is_none() {
                        self.worker_controller
                            .send_message(WorkerMessage::NewWork((pow_hash, target)));
                    }
                }
            }

            // draw progress bar
            {
                self.seals_found += 1;
                self.pb
                    .println(format!("Found! #{} {:#x}", block.number(), block_hash));
                self.pb
                    .set_message(&format!("Total seals found: {:>3}", self.seals_found));
                self.pb.inc(1);
            }
        }
    }
}
