//! Shadow guesser CLI. Predictors run beside the gate; they do not emit tokens.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use nautiburn_histodb::{infer_model, HistDb, ScrapeRecord};
use nautiburn_shadow::{format_report, run_scrape, run_trace};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  nautiburn-shadow score [--db PATH] --scrape FILE [--label NAME]\n  nautiburn-shadow score [--db PATH] --trace FILE [--label NAME] [--model-id ID]\n  nautiburn-shadow score [--db PATH] --scrape FILE --also-b1 TRACE [--label NAME]"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        return usage();
    };
    if cmd != "score" {
        return usage();
    }

    let mut db_path = PathBuf::from("histodb.sqlite");
    let mut scrape: Option<PathBuf> = None;
    let mut trace: Option<PathBuf> = None;
    let mut also_b1: Option<PathBuf> = None;
    let mut label = String::from("shadow");
    let mut model_id = String::from("olmoe-1b-7b-0924");

    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => {
                let Some(p) = args.next() else {
                    eprintln!("--db needs a path");
                    return ExitCode::from(2);
                };
                db_path = PathBuf::from(p);
            }
            "--scrape" => {
                let Some(p) = args.next() else {
                    eprintln!("--scrape needs a JSON file");
                    return ExitCode::from(2);
                };
                scrape = Some(PathBuf::from(p));
            }
            "--trace" => {
                let Some(p) = args.next() else {
                    eprintln!("--trace needs a TSV file");
                    return ExitCode::from(2);
                };
                trace = Some(PathBuf::from(p));
            }
            "--also-b1" => {
                let Some(p) = args.next() else {
                    eprintln!("--also-b1 needs a TSV file");
                    return ExitCode::from(2);
                };
                also_b1 = Some(PathBuf::from(p));
            }
            "--label" => {
                let Some(p) = args.next() else {
                    eprintln!("--label needs a name");
                    return ExitCode::from(2);
                };
                label = p;
            }
            "--model-id" => {
                let Some(p) = args.next() else {
                    eprintln!("--model-id needs an id");
                    return ExitCode::from(2);
                };
                model_id = p;
            }
            other => {
                eprintln!("unknown arg {other}");
                return usage();
            }
        }
    }

    if let Some(parent) = db_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let mut db = match HistDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("open {}: {e}", db_path.display());
            return ExitCode::from(1);
        }
    };

    let mut reports = Vec::new();

    if let Some(path) = scrape {
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("read {}: {e}", path.display());
                return ExitCode::from(1);
            }
        };
        let rec: ScrapeRecord = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("parse {}: {e}", path.display());
                return ExitCode::from(1);
            }
        };
        match run_scrape(&mut db, &rec, &label) {
            Ok(r) => reports.push(r),
            Err(e) => {
                eprintln!("scrape: {e}");
                return ExitCode::from(1);
            }
        }
    }

    let trace_path = trace.or(also_b1);
    if let Some(path) = trace_path {
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("read {}: {e}", path.display());
                return ExitCode::from(1);
            }
        };
        let model = match infer_model(&model_id) {
            Some(m) => m,
            None => {
                eprintln!("unknown model_id {model_id}; upsert is infer-only for olmoe/mixtral");
                return ExitCode::from(1);
            }
        };
        let tlabel = if reports.is_empty() {
            label.clone()
        } else {
            format!("{label}-b1")
        };
        match run_trace(&mut db, &text, "b1-trace", &model, "b1", &tlabel) {
            Ok(r) => reports.push(r),
            Err(e) => {
                eprintln!("trace: {e}");
                return ExitCode::from(1);
            }
        }
    }

    if reports.is_empty() {
        return usage();
    }

    println!("db={}", db_path.display());
    for (i, report) in reports.iter().enumerate() {
        if i > 0 {
            println!();
        }
        print!("{}", format_report(report));
        match serde_json::to_string(report) {
            Ok(j) => println!("json\t{j}"),
            Err(e) => eprintln!("json: {e}"),
        }
    }
    ExitCode::SUCCESS
}
