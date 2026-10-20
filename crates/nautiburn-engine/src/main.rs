use nautiburn_engine::{
    admit_fp16_on_p100, copy_ms, router, router_with_histodb, serve_with_router, shared,
    write_scrape, Calibration, Engine, ExpertPass, ModelProfile, ScrapeDocument,
    CUBECL_F16_SHARED_BYTES, FORCED_TILE_SHARED_BYTES, MIXTRAL_Q4_TWO_EXPERT_BYTES,
    P100_SHARED_CAP_BYTES, TARGET_AHEAD_MS,
};
use nautiburn_histodb::HistDb;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".to_string());
    match cmd.as_str() {
        "fp16-budget" => {
            let stock = admit_fp16_on_p100(CUBECL_F16_SHARED_BYTES);
            let forced = admit_fp16_on_p100(FORCED_TILE_SHARED_BYTES);
            println!(
                "device=p100 compute_dtype=f16 shared_cap={P100_SHARED_CAP_BYTES} cubecl_f16_shared={CUBECL_F16_SHARED_BYTES} stock_launch={stock:?} forced_tile_shared={FORCED_TILE_SHARED_BYTES} forced_launch={forced:?}"
            );
            println!(
                "olmoe_top8_copy_ms={:.4} mixtral_q4_two_expert_copy_ms={:.4} target_ahead_ms={TARGET_AHEAD_MS} basis=planning_12GBps",
                copy_ms(ModelProfile::OLMOE.slab_bytes(8)),
                copy_ms(MIXTRAL_Q4_TWO_EXPERT_BYTES),
            );
            ExitCode::SUCCESS
        }
        "ingest" => {
            let path = args.next().unwrap_or_else(|| "b1/trace.tsv".to_string());
            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    eprintln!("read {path}: {err}");
                    return ExitCode::from(1);
                }
            };
            let mut engine = Engine::new(ModelProfile::OLMOE);
            let n = engine.ingest_trace(&text, "olmoe-b1");
            let prefill: u32 = (0..16)
                .flat_map(|layer| engine.layer_counts("prefill", layer))
                .sum();
            let decode: u32 = (0..16)
                .flat_map(|layer| engine.layer_counts("decode", layer))
                .sum();
            println!("rows={n} prefill_slots={prefill} decode_slots={decode} compute_dtype=f16");
            ExitCode::SUCCESS
        }
        "calibration" => {
            let path = args
                .next()
                .unwrap_or_else(|| "data/olmoe-expert-calibration.json".to_string());
            let text = match fs::read_to_string(&path) {
                Ok(text) => text,
                Err(err) => {
                    eprintln!("read {path}: {err}");
                    return ExitCode::from(1);
                }
            };
            let calibration = match Calibration::from_json(&text) {
                Ok(calibration) => calibration,
                Err(err) => {
                    eprintln!("calibration: {err}");
                    return ExitCode::from(1);
                }
            };
            println!(
                "rows={} source=gguf-expert-weights file={path}",
                calibration.len()
            );
            ExitCode::SUCCESS
        }
        "route" => {
            let trace = args.next();
            let calibration_path = args.next();
            let mut buf = String::new();
            if io::stdin().read_to_string(&mut buf).is_err() {
                eprintln!("route [TRACE] reads one ExpertPass JSON object on stdin");
                return ExitCode::from(1);
            }
            let pass: ExpertPass = match serde_json::from_str(&buf) {
                Ok(pass) => pass,
                Err(err) => {
                    eprintln!("json: {err}");
                    return ExitCode::from(1);
                }
            };
            let mut engine = Engine::new(ModelProfile::OLMOE);
            if let Some(path) = trace {
                let text = match fs::read_to_string(&path) {
                    Ok(text) => text,
                    Err(err) => {
                        eprintln!("read {path}: {err}");
                        return ExitCode::from(1);
                    }
                };
                engine.ingest_trace(&text, &pass.task_class);
            }
            if let Some(path) = calibration_path {
                let text = match fs::read_to_string(&path) {
                    Ok(text) => text,
                    Err(err) => {
                        eprintln!("read {path}: {err}");
                        return ExitCode::from(1);
                    }
                };
                match Calibration::from_json(&text) {
                    Ok(calibration) => engine.set_calibration(calibration),
                    Err(err) => {
                        eprintln!("calibration: {err}");
                        return ExitCode::from(1);
                    }
                }
            }
            let next = engine.route(&pass);
            println!("{}", serde_json::to_string_pretty(&next).unwrap_or_default());
            ExitCode::SUCCESS
        }
        "write-scrape" => {
            let out = args
                .next()
                .unwrap_or_else(|| "/tmp/nautiburn-fp16-scrape/scrape.json".to_string());
            let mut buf = String::new();
            if io::stdin().read_to_string(&mut buf).is_err() {
                eprintln!("write-scrape [OUT] reads one ScrapeDocument JSON object on stdin");
                return ExitCode::from(1);
            }
            let doc: ScrapeDocument = match serde_json::from_str(&buf) {
                Ok(doc) => doc,
                Err(err) => {
                    eprintln!("json: {err}");
                    return ExitCode::from(1);
                }
            };
            if doc.dtype != "f16" {
                eprintln!("dtype must be f16, got {}", doc.dtype);
                return ExitCode::from(2);
            }
            if let Err(err) = write_scrape(&out, &doc) {
                eprintln!("write {out}: {err}");
                return ExitCode::from(1);
            }
            println!("wrote={out} activations={} dtype={}", doc.activations.len(), doc.dtype);
            ExitCode::SUCCESS
        }
        "serve" => {
            // Brief 3. Binds 127.0.0.1:9119 unless a different addr is given.
            // The runtime is built here so the CLI above keeps working exactly
            // as before; this is a new subcommand, not a change to the old ones.
            let addr = args
                .next()
                .unwrap_or_else(|| "127.0.0.1:9119".to_string());
            let sock: std::net::SocketAddr = match addr.parse() {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("serve: bad addr {addr}: {err}");
                    return ExitCode::from(1);
                }
            };
            let state = shared(ModelProfile::OLMOE);
            let mut router = router(state.clone());
            if let Some(flag) = args.next() {
                if flag == "--histodb" {
                    let path = match args.next() {
                        Some(path) => path,
                        None => {
                            eprintln!("serve: --histodb requires a path");
                            return ExitCode::from(2);
                        }
                    };
                    let db = match HistDb::open(path) {
                        Ok(db) => db,
                        Err(err) => {
                            eprintln!("serve: histodb: {err}");
                            return ExitCode::from(1);
                        }
                    };
                    router = router_with_histodb(state, Some(Arc::new(Mutex::new(db))));
                } else {
                    eprintln!("serve: unexpected argument {flag}");
                    return ExitCode::from(2);
                }
            }
            eprintln!("nautiburn-engine listening on http://{sock}");
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(err) => {
                    eprintln!("serve: tokio runtime: {err}");
                    return ExitCode::from(1);
                }
            };
            match rt.block_on(serve_with_router(sock, router)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("serve: {err}");
                    ExitCode::from(1)
                }
            }
        }
        _ => {
            eprintln!(
                "commands: fp16-budget | ingest TRACE | calibration [JSON] | route [TRACE] [CALIBRATION] < pass.json | write-scrape [OUT] < scrape.json | serve [ADDR]"
            );
            ExitCode::from(2)
        }
    }
}
