use ckb_build_info::Version;
use ckb_logger::info_target;
use sentry::{
    configure_scope, init,
    integrations::panic::register_panic_handler,
    internals::{ClientInitGuard, Dsn},
    protocol::Event,
    ClientOptions, Level,
};
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SentryConfig {
    pub dsn: String,
    pub org_ident: Option<String>,
    pub org_contact: Option<String>,
}

impl SentryConfig {
    pub fn is_enabled(&self) -> bool {
        self.dsn.parse::<Dsn>().is_ok()
    }
}

pub fn sentry_init(config: &SentryConfig, version: &Version) -> Option<ClientInitGuard> {
    if config.is_enabled() {
        info_target!(
            "sentry",
            "**Notice**: \
             The ckb process will send stack trace to sentry on Rust panics. \
             This is enabled by default before mainnet, which can be opted out by setting \
             the option `dsn` to empty in the config file. The DSN is now {}",
            config.dsn
        );

        let guard = init(build_sentry_client_options(&config, &version));
        if guard.is_enabled() {
            configure_scope(|scope| {
                scope.set_tag("release.pre", version.is_pre());
                scope.set_tag("release.dirty", version.is_dirty());
                scope.set_tag("subcommand", "miner");
                if let Some(org_ident) = &config.org_ident {
                    scope.set_tag("org_ident", org_ident);
                }
                if let Some(org_contact) = &config.org_contact {
                    scope.set_extra("org_contact", org_contact.clone().into());
                }
            });

            register_panic_handler();
        }

        Some(guard)
    } else {
        None
    }
}

fn build_sentry_client_options(config: &SentryConfig, version: &Version) -> ClientOptions {
    ClientOptions {
        dsn: config.dsn.parse().ok(),
        release: Some(version.long().into()),
        before_send: Some(Arc::new(Box::new(before_send))),
        ..Default::default()
    }
}

static DB_OPEN_FINGERPRINT: &[Cow<'static, str>] =
    &[Cow::Borrowed("ckb-db"), Cow::Borrowed("open")];
static SQLITE_FINGERPRINT: &[Cow<'static, str>] = &[
    Cow::Borrowed("ckb-network"),
    Cow::Borrowed("peerstore"),
    Cow::Borrowed("sqlite"),
];

fn before_send(mut event: Event<'static>) -> Option<Event<'static>> {
    let ex = match event
        .exception
        .values
        .iter()
        .next()
        .and_then(|ex| ex.value.as_ref())
    {
        Some(ex) => ex,
        None => return Some(event),
    };

    // Group events via fingerprint, or ignore

    if ex.starts_with("DBError failed to open the database") {
        event.level = Level::Warning;
        event.fingerprint = Cow::Borrowed(DB_OPEN_FINGERPRINT);
    } else if ex.contains("SqliteFailure") {
        event.level = Level::Warning;
        event.fingerprint = Cow::Borrowed(SQLITE_FINGERPRINT);
    } else if ex.starts_with("DBError the database version")
        || ex.contains("kind: AddrInUse")
        || ex.contains("kind: AddrNotAvailable")
        || ex.contains("IO error: No space left")
    {
        // ignore
        return None;
    }

    Some(event)
}
