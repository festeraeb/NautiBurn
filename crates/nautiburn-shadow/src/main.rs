//! Shadow guesser CLI. Predictors run beside the gate; they do not emit tokens.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use nautiburn_histodb::{infer_model, HistDb, ScrapeRecord};
use nautiburn_shadow::{
    events_from_scrape, format_next_expert_report, format_report, persist_next_report,
    report_next_expert, run_scrape, run_scrape_next, run_trace,
};

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  nautiburn-shadow score [--db PATH] --scrape FILE [--also-decode FILE] [--label NAME]\n  nautiburn-shadow score [--db PATH] --trace FILE [--label NAME] [--model-id ID]\n  nautiburn-shadow score [--db PATH] --scrape FILE --also-b1 TRACE [--label NAME]\n  add --same-step to also print the old current-gate table"
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
    let mut also_decode: Option<PathBuf> = None;
    let mut trace: Option<PathBuf> = None;
    let mut also_b1: Option<PathBuf> = None;
    let mut label = String::from("shadow");
    let mut model_id = String::from("olmoe-1b-7b-0924");
    let mut also_same_step = false;

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
            "--also-decode" => {
                let Some(p) = args.next() else {
                    eprintln!("--also-decode needs a JSON file");
                    return ExitCode::from(2);
                };
                also_decode = Some(PathBuf::from(p));
            }
            "--same-step" => also_same_step = true,
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

    let mut same_step_reports = Vec::new();
    let mut printed_next = false;

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
        match run_scrape_next(&mut db, &rec, &label) {
            Ok(r) => {
                println!("db={}", db_path.display());
                print!("{}", format_next_expert_report(&r));
                match serde_json::to_string(&r) {
                    Ok(j) => println!("json\t{j}"),
                    Err(e) => eprintln!("json: {e}"),
                }
                printed_next = true;
            }
            Err(e) => {
                eprintln!("scrape: {e}");
                return ExitCode::from(1);
            }
        }
        if also_same_step {
            match run_scrape(&mut db, &rec, &format!("{label}-same-step")) {
                Ok(r) => same_step_reports.push(r),
                Err(e) => {
                    eprintln!("same-step: {e}");
                    return ExitCode::from(1);
                }
            }
        }

        if let Some(dpath) = also_decode {
            let dtext = match fs::read_to_string(&dpath) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("read {}: {e}", dpath.display());
                    return ExitCode::from(1);
                }
            };
            let drec: ScrapeRecord = match serde_json::from_str(&dtext) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("parse {}: {e}", dpath.display());
                    return ExitCode::from(1);
                }
            };
            if db.model(&drec.model_id).ok().flatten().is_none() {
                if let Some(m) = infer_model(&drec.model_id) {
                    let _ = db.upsert_model(&m);
                }
            }
            if let Err(e) = db.insert_scrape(&drec) {
                eprintln!("decode ingest: {e}");
                return ExitCode::from(1);
            }
            let mut combined = events_from_scrape(&rec);
            combined.extend(events_from_scrape(&drec));
            let model = match db.model(&rec.model_id) {
                Ok(Some(m)) => m,
                _ => {
                    eprintln!("missing model after ingest");
                    return ExitCode::from(1);
                }
            };
            if let Err(e) = persist_next_report(
                &mut db,
                &drec.request_id,
                &events_from_scrape(&drec),
                model.n_layers as usize,
                model.n_experts as usize,
                model.top_k as usize,
            ) {
                eprintln!("decode persist: {e}");
                return ExitCode::from(1);
            }
            let combined_report = report_next_expert(
                &format!("{label}-combined"),
                &format!("{}+{}", rec.request_id, drec.request_id),
                &rec.model_id,
                "scrape",
                &combined,
                model.n_layers as usize,
                model.n_experts as usize,
                model.top_k as usize,
            );
            println!();
            print!("{}", format_next_expert_report(&combined_report));
            match serde_json::to_string(&combined_report) {
                Ok(j) => println!("json\t{j}"),
                Err(e) => eprintln!("json: {e}"),
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
        let tlabel = if printed_next {
            format!("{label}-b1")
        } else {
            label.clone()
        };
        let events = nautiburn_shadow::events_from_trace(&text);
        if let Err(e) = db.upsert_model(&model) {
            eprintln!("upsert model: {e}");
            return ExitCode::from(1);
        }
        let n_tokens = events.iter().map(|e| e.token_pos).max().map(|m| m + 1);
        if let Err(e) = db.upsert_request(
            "b1-trace",
            &model.model_id,
            "b1",
            "b1_trace",
            Some("mixed"),
            n_tokens,
            Some("f16"),
        ) {
            eprintln!("upsert request: {e}");
            return ExitCode::from(1);
        }
        if let Err(e) = persist_next_report(
            &mut db,
            "b1-trace",
            &events,
            model.n_layers as usize,
            model.n_experts as usize,
            model.top_k as usize,
        ) {
            eprintln!("trace persist: {e}");
            return ExitCode::from(1);
        }
        let next = report_next_expert(
            &tlabel,
            "b1-trace",
            &model.model_id,
            "b1_trace",
            &events,
            model.n_layers as usize,
            model.n_experts as usize,
            model.top_k as usize,
        );
        if printed_next {
            println!();
        } else {
            println!("db={}", db_path.display());
        }
        print!("{}", format_next_expert_report(&next));
        match serde_json::to_string(&next) {
            Ok(j) => println!("json\t{j}"),
            Err(e) => eprintln!("json: {e}"),
        }
        printed_next = true;
        if also_same_step {
            match run_trace(&mut db, &text, "b1-trace-same-step", &model, "b1", &format!("{tlabel}-same-step")) {
                Ok(r) => same_step_reports.push(r),
                Err(e) => {
                    eprintln!("trace same-step: {e}");
                    return ExitCode::from(1);
                }
            }
        }
    }

    for report in &same_step_reports {
        println!();
        print!("{}", format_report(report));
        match serde_json::to_string(report) {
            Ok(j) => println!("json\t{j}"),
            Err(e) => eprintln!("json: {e}"),
        }
    }

    if !printed_next && same_step_reports.is_empty() {
        return usage();
    }
    ExitCode::SUCCESS
}
