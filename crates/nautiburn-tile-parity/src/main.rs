use std::env;
use std::process::exit;
use std::time::Instant;

use nautiburn_tile_parity::{compare, gemm_cpu_f32, gemm_f16, random_matrix, Backend};

fn json_num(x: f32) -> String {
    if x.is_finite() {
        format!("{:.6}", x)
    } else {
        "1e30".to_string()
    }
}

fn parse_args() -> (Backend, String, usize, usize, usize, u64, f32, f32, usize) {
    let mut backend = None;
    let mut device = None;
    let mut wgpu_kind = None;
    let mut m = 64usize;
    let mut n = 2048usize;
    let mut k = 1024usize;
    let mut seed = 7u64;
    let mut atol = 0.05f32;
    let mut rtol = 0.02f32;
    let mut iters = 5usize;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        i += 1;

        match arg {
            "--backend" => {
                let value = args.get(i).ok_or("missing value for --backend").unwrap();
                i += 1;
                backend = Some(value.clone());
            }
            "--device" => {
                let value = args.get(i).ok_or("missing value for --device").unwrap();
                i += 1;
                device = Some(value.parse::<usize>().unwrap());
            }
            "--wgpu-kind" => {
                let value = args.get(i).ok_or("missing value for --wgpu-kind").unwrap();
                i += 1;
                wgpu_kind = Some(value.clone());
            }
            "--m" => {
                let value = args.get(i).ok_or("missing value for --m").unwrap();
                i += 1;
                m = value.parse::<usize>().unwrap();
            }
            "--n" => {
                let value = args.get(i).ok_or("missing value for --n").unwrap();
                i += 1;
                n = value.parse::<usize>().unwrap();
            }
            "--k" => {
                let value = args.get(i).ok_or("missing value for --k").unwrap();
                i += 1;
                k = value.parse::<usize>().unwrap();
            }
            "--seed" => {
                let value = args.get(i).ok_or("missing value for --seed").unwrap();
                i += 1;
                seed = value.parse::<u64>().unwrap();
            }
            "--atol" => {
                let value = args.get(i).ok_or("missing value for --atol").unwrap();
                i += 1;
                atol = value.parse::<f32>().unwrap();
            }
            "--rtol" => {
                let value = args.get(i).ok_or("missing value for --rtol").unwrap();
                i += 1;
                rtol = value.parse::<f32>().unwrap();
            }
            "--iters" => {
                let value = args.get(i).ok_or("missing value for --iters").unwrap();
                i += 1;
                iters = value.parse::<usize>().unwrap();
            }
            other => {
                panic!("unknown argument: {other}");
            }
        }
    }

    let backend_name = backend.unwrap_or_else(|| "cuda".to_string());
    let device_name = if backend_name == "cuda" {
        format!("{}", device.unwrap_or(0))
    } else {
        wgpu_kind.unwrap_or_else(|| "default".to_string())
    };

    let backend = match backend_name.as_str() {
        "cuda" => Backend::Cuda(device.unwrap_or(0)),
        "wgpu" => Backend::Wgpu(device_name.clone()),
        other => panic!("backend must be cuda or wgpu, got {other}"),
    };

    (backend, device_name, m, n, k, seed, atol, rtol, iters)
}

fn median(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn main() {
    let (backend, device_name, m, n, k, seed, atol, rtol, iters) = parse_args();

    let client = match nautiburn_tile_parity::open(&backend) {
        Ok(client) => client,
        Err(err) => {
            println!(
                r#"{{"backend":"{}","device":"{}","m":{},"n":{},"k":{},"seed":{},"max_abs_err":0.000000,"mean_abs_err":0.000000,"max_rel_err":0.000000,"n_bad":0,"median_ms":0.000000,"gflops":0.000000,"pass":false}}"#,
                backend_name(&backend),
                device_name,
                m,
                n,
                k,
                seed
            );
            eprintln!("backend open failed: {err}");
            exit(2);
        }
    };

    let lhs = random_matrix(m, k, seed);
    let rhs = random_matrix(k, n, seed.wrapping_add(1));

    let warm = gemm_f16(&client, &lhs, &rhs, m, n, k);
    if warm.is_err() {
        println!(
            r#"{{"backend":"{}","device":"{}","m":{},"n":{},"k":{},"seed":{},"max_abs_err":0.000000,"mean_abs_err":0.000000,"max_rel_err":0.000000,"n_bad":0,"median_ms":0.000000,"gflops":0.000000,"pass":false}}"#,
            backend_name(&backend),
            device_name,
            m,
            n,
            k,
            seed
        );
        eprintln!("backend gemm failed: {}", warm.unwrap_err());
        exit(2);
    }

    let mut durations = Vec::new();
    let mut last = warm.unwrap();

    for _ in 0..iters {
        let start = Instant::now();
        last = match gemm_f16(&client, &lhs, &rhs, m, n, k) {
            Ok(out) => out,
            Err(err) => {
                println!(
                    r#"{{"backend":"{}","device":"{}","m":{},"n":{},"k":{},"seed":{},"max_abs_err":0.000000,"mean_abs_err":0.000000,"max_rel_err":0.000000,"n_bad":0,"median_ms":0.000000,"gflops":0.000000,"pass":false}}"#,
                    backend_name(&backend),
                    device_name,
                    m,
                    n,
                    k,
                    seed
                );
                eprintln!("backend gemm failed: {err}");
                exit(2);
            }
        };
        durations.push(start.elapsed().as_secs_f64());
    }

    let want = gemm_cpu_f32(&lhs, &rhs, m, n, k);
    let parity = compare(&last, &want, atol, rtol);

    let median_seconds = median(&mut durations);
    let median_ms = median_seconds * 1000.0;
    let gflops = if median_seconds > 0.0 {
        (2.0 * m as f64 * n as f64 * k as f64) / median_seconds / 1.0e9
    } else {
        0.0
    };

    let pass = parity.n_bad == 0;

    println!(
        r#"{{"backend":"{}","device":"{}","m":{},"n":{},"k":{},"seed":{},"max_abs_err":{},"mean_abs_err":{},"max_rel_err":{},"n_bad":{},"median_ms":{:.6},"gflops":{:.6},"pass":{}}}"#,
        backend_name(&backend),
        device_name,
        m,
        n,
        k,
        seed,
        json_num(parity.max_abs_err),
        json_num(parity.mean_abs_err),
        json_num(parity.max_rel_err),
        parity.n_bad,
        median_ms,
        gflops,
        pass
    );

    if pass {
        exit(0);
    } else {
        exit(1);
    }
}

fn backend_name(backend: &Backend) -> String {
    match backend {
        Backend::Cuda(_) => "cuda".to_string(),
        Backend::Wgpu(_) => "wgpu".to_string(),
    }
}
