use std::io;

/// Uses 0, 64 - 113 as exit code.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExitCode {
    Config = 65,
    IO = 66,
}

impl From<io::Error> for ExitCode {
    fn from(err: io::Error) -> ExitCode {
        eprintln!("IO Error: {:?}", err);
        ExitCode::IO
    }
}

impl From<toml::de::Error> for ExitCode {
    fn from(err: toml::de::Error) -> ExitCode {
        eprintln!("Config Error: {:?}", err);
        ExitCode::Config
    }
}
