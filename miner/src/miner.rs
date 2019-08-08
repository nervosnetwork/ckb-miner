use crate::client::Client;
use crate::worker::{start_worker, WorkerController, WorkerMessage};
use crate::MinerConfig;
use crate::Work;
use ckb_core::block::BlockBuilder;
use ckb_core::difficulty::difficulty_to_target;
use ckb_core::header::Seal;
use ckb_logger::{debug, error, info};
use ckb_util::Mutex;
use crossbeam_channel::{unbounded, Receiver};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use lru_cache::LruCache;
use numext_fixed_hash::H256;
use std::sync::Arc;
use std::thread;
use std::time;

const WORK_CACHE_SIZE: usize = 32;

pub struct Miner {
    pub client: Client,
    pub works: Arc<Mutex<LruCache<H256, Work>>>,
    pub worker_controller: WorkerController,
    pub seal_rx: Receiver<(H256, Seal)>,
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
                    let pow_hash = work.block.header().pow_hash();
                    let target = difficulty_to_target(&work.block.header().difficulty());
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
                Ok((pow_hash, seal)) => self.check_seal(pow_hash, seal),
                _ => {
                    error!("seal_rx closed");
                    break;
                }
            }
        }
    }

    fn check_seal(&mut self, pow_hash: H256, seal: Seal) {
        let new_work = { self.works.lock().get_refresh(&pow_hash).cloned() };
        if let Some(work) = new_work {
            let raw_header = work.block.header().raw().to_owned();
            let block = BlockBuilder::from_block(work.block)
                .header(raw_header.with_seal(seal))
                .build();

            if self.stderr_is_tty {
                debug!(
                    "Found! #{} {:#x}",
                    block.header().number(),
                    block.header().hash(),
                );
            } else {
                info!(
                    "Found! #{} {:#x}",
                    block.header().number(),
                    block.header().hash(),
                );
            }

            // submit block and poll new work
            {
                self.client.submit_block(&work.work_id.to_string(), &block);
                // poll new work
                if let Some(work) = self.client.poll_new_work() {
                    let pow_hash = work.block.header().pow_hash();
                    let target = difficulty_to_target(&work.block.header().difficulty());
                    if self.works.lock().insert(pow_hash.clone(), work).is_none() {
                        self.worker_controller
                            .send_message(WorkerMessage::NewWork((pow_hash, target)));
                    }
                }
            }

            // draw progress bar
            {
                self.seals_found += 1;
                self.pb.println(format!(
                    "Found! #{} {:#x}",
                    block.header().number(),
                    block.header().hash()
                ));
                self.pb
                    .set_message(&format!("Total seals found: {:>3}", self.seals_found));
                self.pb.inc(1);
            }
        }
    }
}
