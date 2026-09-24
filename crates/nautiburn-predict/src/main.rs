//! Print the five dumb expert-predictor rows.
//!
//! Defaults read `b1/trace.tsv` and `b1/public-slice.tsv` from the current
//! directory. Pass those two paths to score another pair.

use std::env;
use std::fs;
use std::process::ExitCode;

use nautiburn_predict::{evaluate, format_report};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let b1_path = args.next().unwrap_or_else(|| "b1/trace.tsv".to_string());
    let public_path = args.next().unwrap_or_else(|| "b1/public-slice.tsv".to_string());
    let b1 = match fs::read_to_string(&b1_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("read {b1_path}: {err}");
            return ExitCode::from(1);
        }
    };
    let public = match fs::read_to_string(&public_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("read {public_path}: {err}");
            return ExitCode::from(1);
        }
    };
    let report = evaluate(&b1, &public);
    print!("{}", format_report(&report));
    ExitCode::from(0)
}
