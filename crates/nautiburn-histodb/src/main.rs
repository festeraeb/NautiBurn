//! Thin CLI: `nautiburn-histodb ingest scrape.json` and `ingest --allenai-olmoe dir/`.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use nautiburn_histodb::HistDb;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  nautiburn-histodb ingest [--db PATH] scrape.json [more.json ...]   (several: one ok/err line each)\n  nautiburn-histodb ingest [--db PATH] --allenai-olmoe PATH\n  nautiburn-histodb ingest [--db PATH] --allenai-mixtral PATH\n  nautiburn-histodb ingest [--db PATH] --continuum-counts FILE"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        return usage();
    };
    if cmd != "ingest" {
        return usage();
    }

    let mut db_path = PathBuf::from("histodb.sqlite");
    let mut allenai_olmoe: Option<PathBuf> = None;
    let mut allenai_mixtral: Option<PathBuf> = None;
    let mut continuum: Option<PathBuf> = None;
    let mut scrapes: Vec<PathBuf> = Vec::new();

    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => {
                let Some(p) = args.next() else {
                    eprintln!("--db needs a path");
                    return ExitCode::from(2);
                };
                db_path = PathBuf::from(p);
            }
            "--allenai-olmoe" => {
                let Some(p) = args.next() else {
                    eprintln!("--allenai-olmoe needs a JSONL file or directory");
                    return ExitCode::from(2);
                };
                allenai_olmoe = Some(PathBuf::from(p));
            }
            "--allenai-mixtral" => {
                let Some(p) = args.next() else {
                    eprintln!("--allenai-mixtral needs a JSONL file or directory");
                    return ExitCode::from(2);
                };
                allenai_mixtral = Some(PathBuf::from(p));
            }
            "--continuum-counts" => {
                let Some(p) = args.next() else {
                    eprintln!("--continuum-counts needs a JSON file");
                    return ExitCode::from(2);
                };
                continuum = Some(PathBuf::from(p));
            }
            other if other.starts_with('-') => {
                eprintln!("unknown flag {other}");
                return usage();
            }
            other => scrapes.push(PathBuf::from(other)),
        }
    }

    // Many scrape files in one call: the DB is opened (and counted) once per batch instead of once
    // per file, which is what kept the consume loop near 77 files/min. One line per file.
    if scrapes.len() > 1 {
        if allenai_olmoe.is_some() || allenai_mixtral.is_some() || continuum.is_some() {
            eprintln!("several scrape files cannot be combined with another import");
            return usage();
        }
        return match ingest_many(&db_path, &scrapes) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(_) => ExitCode::from(1),
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        };
    }
    let scrape = scrapes.pop();
    if let Err(e) = run(db_path, scrape, allenai_olmoe, allenai_mixtral, continuum) {
        eprintln!("{e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Ingest each file, printing `ok <path>` or `err <path> <error>`; returns how many failed.
fn ingest_many(db_path: &PathBuf, paths: &[PathBuf]) -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(parent) = db_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut db = HistDb::open(db_path)?;
    let mut failed = 0;
    for p in paths {
        match db.ingest_scrape_file(p, None) {
            Ok(_) => println!("ok {}", p.display()),
            Err(e) => {
                failed += 1;
                println!("err {} {}", p.display(), e.to_string().replace('\n', " "));
            }
        }
    }
    Ok(failed)
}

fn run(
    db_path: PathBuf,
    scrape: Option<PathBuf>,
    allenai_olmoe: Option<PathBuf>,
    allenai_mixtral: Option<PathBuf>,
    continuum: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = db_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut db = HistDb::open(&db_path)?;
    let before = db.histogram_row_count()?;
    println!("db={}", db_path.display());
    println!("table=histogram_counts");
    println!("histogram_counts_before={before}");

    match (scrape, allenai_olmoe, allenai_mixtral, continuum) {
        (Some(path), None, None, None) => {
            let rec = db.ingest_scrape_file(&path, None)?;
            println!("ingested={}", path.display());
            println!("request_id={}", rec.request_id);
            println!("model_id={}", rec.model_id);
            match db.request_dtype(&rec.request_id)? {
                Some(dt) => println!("dtype={dt}"),
                None => println!("dtype="),
            }
        }
        (None, Some(dir), None, None) => {
            let n = db.import_allenai_olmoe(&dir)?;
            println!("ingested_allenai_olmoe={} rows={n}", dir.display());
        }
        (None, None, Some(dir), None) => {
            let n = db.import_allenai_mixtral(&dir)?;
            println!("ingested_allenai_mixtral={} rows={n}", dir.display());
        }
        (None, None, None, Some(path)) => {
            let n = db.import_continuum_counts(&path)?;
            println!("ingested_continuum_counts={} cells={n}", path.display());
        }
        _ => {
            return Err("need exactly one of scrape.json | --allenai-olmoe PATH | --allenai-mixtral PATH | --continuum-counts FILE".into());
        }
    }

    let after = db.histogram_row_count()?;
    println!("histogram_counts_after={after}");
    if let Some(row) = db.histogram_sample()? {
        println!(
            "sample_row model_id={} class_id={} phase={} layer={} expert_id={} count={}",
            row.model_id, row.class_id, row.phase, row.layer, row.expert_id, row.count
        );
    } else {
        println!("sample_row=none");
    }
    Ok(())
}
