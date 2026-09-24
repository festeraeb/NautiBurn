use nautiburn_engine::{
    admit_fp16_on_p100, CUBECL_F16_SHARED_BYTES, Engine, ExpertPass, ModelProfile,
    P100_SHARED_CAP_BYTES, MIXTRAL_Q4_TWO_EXPERT_BYTES, copy_ms, TARGET_AHEAD_MS,
};
use std::env;
use std::fs;
use std::io::{self, Read};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".to_string());
    match cmd.as_str() {
        "fp16-budget" => {
            let launch = admit_fp16_on_p100(CUBECL_F16_SHARED_BYTES);
            println!(
                "device=p100 compute_dtype=f16 shared_cap={P100_SHARED_CAP_BYTES} cubecl_f16_shared={CUBECL_F16_SHARED_BYTES} launch={launch:?}"
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
        "route" => {
            let trace = args.next();
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
            let next = engine.route(&pass);
            println!("{}", serde_json::to_string_pretty(&next).unwrap_or_default());
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("commands: fp16-budget | ingest TRACE | route [TRACE] < pass.json");
            ExitCode::from(2)
        }
    }
}
