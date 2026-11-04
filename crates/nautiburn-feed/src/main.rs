//! nautiburn-feed: send varied prompts through live MoE seats, then tag their scrape rows.
//!
//! Live llama-server scrapes carry no prompt: a row is `nb-server-<unix second>-<n>` with the
//! seat's model_id and class `live-<seat>`. The feeder sends each seat ONE request at a time,
//! with a gap between requests, and records the second window each request ran in. `tag` then
//! sets `task_domain` and `prompt_text` on every row of that model whose second falls in the
//! window. Rows from other traffic in the same window would be mis-tagged, so run it while the
//! forge queue is idle, and windows are kept tight (+/- 1 s) with a gap of `--gap-s` between them.
//!
//!   nautiburn-feed run --seats seats.json --prompts prompts.jsonl --ledger ledger.jsonl [--rounds 1]
//!                      [--max-tokens 256] [--gap-s 4]
//!   nautiburn-feed tag --db sidechannel.sqlite --ledger ledger.jsonl [--min-age-s 120] [--every S]

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Deserialize, Clone)]
struct Seat {
    name: String,
    url: String,
    model_id: String,
}

#[derive(Deserialize, Clone)]
struct Prompt {
    domain: String,
    prompt: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Entry {
    pub id: String,
    pub seat: String,
    pub model_id: String,
    pub domain: String,
    pub prompt: String,
    pub t0: i64,
    pub t1: i64,
    pub ok: bool,
    pub completion_tokens: Option<i64>,
    pub error: Option<String>,
}

fn now_s() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &str) -> Vec<T> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    BufReader::new(f)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect()
}

/// The rows a ledger entry owns: same model, `nb-server-<second>` inside [t0, t1].
pub const TAG_SQL: &str = "UPDATE requests SET task_domain = ?1, prompt_text = ?2 \
     WHERE model_id = ?3 AND request_id LIKE 'nb-server-%' AND prompt_text IS NULL \
     AND CAST(substr(request_id, 11, instr(substr(request_id, 11), '-') - 1) AS INTEGER) BETWEEN ?4 AND ?5";

pub fn tag_entry(db: &rusqlite::Connection, e: &Entry) -> rusqlite::Result<usize> {
    db.execute(TAG_SQL, rusqlite::params![e.domain, e.prompt, e.model_id, e.t0, e.t1])
}

fn run(args: &[String]) {
    let seats: Vec<Seat> = serde_json::from_str(
        &std::fs::read_to_string(arg(args, "--seats").expect("--seats")).expect("read seats"),
    )
    .expect("seats json");
    let prompts: Vec<Prompt> = read_jsonl(&arg(args, "--prompts").expect("--prompts"));
    let ledger_path = arg(args, "--ledger").expect("--ledger");
    let rounds: usize = arg(args, "--rounds").and_then(|v| v.parse().ok()).unwrap_or(1);
    let max_tokens: u32 = arg(args, "--max-tokens").and_then(|v| v.parse().ok()).unwrap_or(256);
    let gap = Duration::from_secs(arg(args, "--gap-s").and_then(|v| v.parse().ok()).unwrap_or(4));
    assert!(!prompts.is_empty(), "no prompts");
    let ledger = Arc::new(Mutex::new(
        OpenOptions::new().create(true).append(true).open(&ledger_path).expect("open ledger"),
    ));
    let mut workers = Vec::new();
    for (si, seat) in seats.into_iter().enumerate() {
        let prompts = prompts.clone();
        let ledger = ledger.clone();
        workers.push(std::thread::spawn(move || {
            let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(600)).build();
            let n = prompts.len();
            for r in 0..rounds {
                // Each seat walks the list from a different start so seats cover domains in a different order.
                for k in 0..n {
                    let p = &prompts[(k + si * 37 + r * 11) % n];
                    let t0 = now_s().floor() as i64 - 1;
                    let body = serde_json::json!({
                        "messages": [{"role": "user", "content": p.prompt}],
                        "max_tokens": max_tokens, "temperature": 0.7, "stream": false,
                    });
                    let res = agent.post(&format!("{}/v1/chat/completions", seat.url)).send_json(body);
                    let t1 = now_s().ceil() as i64 + 1;
                    let (ok, toks, err) = match res {
                        Ok(resp) => match resp.into_json::<serde_json::Value>() {
                            Ok(v) => (true, v["usage"]["completion_tokens"].as_i64(), None),
                            Err(e) => (false, None, Some(e.to_string())),
                        },
                        Err(e) => (false, None, Some(e.to_string())),
                    };
                    let e = Entry {
                        id: format!("{}-{}-{}", seat.name, t0, k),
                        seat: seat.name.clone(),
                        model_id: seat.model_id.clone(),
                        domain: p.domain.clone(),
                        prompt: p.prompt.clone(),
                        t0,
                        t1,
                        ok,
                        completion_tokens: toks,
                        error: err,
                    };
                    {
                        let mut f = ledger.lock().unwrap();
                        let _ = writeln!(f, "{}", serde_json::to_string(&e).unwrap());
                    }
                    eprintln!("[{}] {} {} ok={} toks={:?} {}s", seat.name, e.domain, k, ok, toks, t1 - t0);
                    std::thread::sleep(gap);
                }
            }
        }));
    }
    for w in workers {
        let _ = w.join();
    }
}

fn tag(args: &[String]) {
    let db_path = arg(args, "--db").expect("--db");
    let ledger_path = arg(args, "--ledger").expect("--ledger");
    let min_age: i64 = arg(args, "--min-age-s").and_then(|v| v.parse().ok()).unwrap_or(120);
    let every: Option<u64> = arg(args, "--every").and_then(|v| v.parse().ok());
    let done_path = format!("{ledger_path}.tagged");
    loop {
        let done: std::collections::HashSet<String> = std::fs::read_to_string(&done_path)
            .unwrap_or_default()
            .lines()
            .map(|l| l.split('\t').next().unwrap_or("").to_string())
            .collect();
        let db = rusqlite::Connection::open(&db_path).expect("open db");
        db.busy_timeout(Duration::from_secs(60)).unwrap();
        let mut out = OpenOptions::new().create(true).append(true).open(&done_path).unwrap();
        let (mut entries, mut rows) = (0, 0);
        let cutoff = now_s() as i64 - min_age;
        for e in read_jsonl::<Entry>(&ledger_path) {
            if !e.ok || done.contains(&e.id) || e.t1 > cutoff {
                continue;
            }
            match tag_entry(&db, &e) {
                Ok(n) if n > 0 => {
                    entries += 1;
                    rows += n;
                    let _ = writeln!(out, "{}\t{}", e.id, n);
                }
                Ok(_) => {} // rows not ingested yet; retried next pass
                Err(err) => eprintln!("tag {}: {err}", e.id),
            }
        }
        eprintln!("tagged {entries} ledger entries, {rows} rows");
        match every {
            Some(s) => std::thread::sleep(Duration::from_secs(s)),
            None => break,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => run(&args),
        Some("tag") => tag(&args),
        _ => {
            eprintln!("usage: nautiburn-feed run|tag ... (see the module doc)");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> rusqlite::Connection {
        let db = rusqlite::Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE requests (request_id TEXT PRIMARY KEY, model_id TEXT, class_id TEXT,
             task_domain TEXT, prompt_text TEXT);",
        )
        .unwrap();
        for (id, model) in [
            ("nb-server-1793819619-2", "kat"),   // before the window
            ("nb-server-1793819620-10", "kat"),  // inside
            ("nb-server-1793819622-11", "kat"),  // inside
            ("nb-server-1793819623-12", "kat"),  // after
            ("nb-server-1793819621-3", "flint"), // inside, other model
            ("run-20261104T204707-x", "kat"),    // not a live row
        ] {
            db.execute("INSERT INTO requests VALUES (?1, ?2, 'live', NULL, NULL)", [id, model]).unwrap();
        }
        db
    }

    fn entry() -> Entry {
        Entry {
            id: "kat-1".into(),
            seat: "kat".into(),
            model_id: "kat".into(),
            domain: "art".into(),
            prompt: "Describe Impressionism.".into(),
            t0: 1793819620,
            t1: 1793819622,
            ok: true,
            completion_tokens: Some(10),
            error: None,
        }
    }

    #[test]
    fn tags_only_this_models_rows_inside_the_window() {
        let db = db();
        assert_eq!(tag_entry(&db, &entry()).unwrap(), 2);
        let tagged: Vec<String> = db
            .prepare("SELECT request_id FROM requests WHERE task_domain = 'art' ORDER BY request_id")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(tagged, vec!["nb-server-1793819620-10", "nb-server-1793819622-11"]);
    }

    #[test]
    fn never_overwrites_a_row_that_already_has_a_prompt() {
        let db = db();
        assert_eq!(tag_entry(&db, &entry()).unwrap(), 2);
        let mut other = entry();
        other.domain = "politics".into();
        assert_eq!(tag_entry(&db, &other).unwrap(), 0);
    }
}
