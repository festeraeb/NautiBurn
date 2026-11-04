//! Bridge: sidechannel scrapes -> the PAMP ensemble's telemetry stream.
//!
//! # Why this exists
//!
//! Expert routing and PAMP have been two disconnected streams. Everything the
//! scraper captures lands in `sidechannel.sqlite` and feeds /Nauti-Burn, while
//! PAMP's own log carries task outcomes and essentially nothing else. Measured
//! 2026-11-04 across 51,099 records in `pamp_telemetry.jsonl`:
//!
//! | kind | records |
//! |---|---|
//! | `role_outcome` | 51,092 |
//! | `pamp_plan` | 4 |
//! | `histogram_observation` | 4 |
//! | `moe_route` | 2 |
//! | `moe_prediction` | 1 |
//!
//! The ensemble has the routing kinds defined and no data in them, so it can
//! model which *model* succeeded at a task but never which *experts* ran. This
//! exports the rows it has been missing, from the database that already holds
//! them. No engine changes, no re-scrape.
//!
//! # What is exported, and what is deliberately not
//!
//! `moe_route` per (request, phase, layer, token position): the ranked expert
//! ids actually selected, with the prompt identity and forge provenance that
//! tonight's schema change added.
//!
//! `expert_weights` is emitted as **null**, not as a number derived from rank.
//! The scraper records the top-k *ordering*, never the gate weights, so any
//! weight here would be invented. A null the ensemble can skip is worth more
//! than a plausible value it would train on.
//!
//! Export is watermarked by `request_id`, so re-running is idempotent and a
//! row is never emitted twice.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use rusqlite::Connection;

use crate::schema::HistDbError;

/// Where the watermark lives inside the histodb, so it travels with the data.
const WATERMARK_KEY: &str = "pamp_export_after";

/// One `moe_route` payload, matching the schema already in pamp_telemetry.jsonl.
#[derive(serde::Serialize)]
struct MoeRoute<'a> {
    run_id: &'a str,
    model_id: &'a str,
    task_class: Option<&'a str>,
    layer: i64,
    token_index: i64,
    expert_ids: Vec<i64>,
    /// Always null: the scraper records rank, never gate weight. See module docs.
    expert_weights: Option<Vec<f64>>,
    expert_logits: Option<Vec<f64>>,
    token_id: Option<i64>,
    timestamp_ns: i64,
    // fields added by the 2026-11-04 prompt-awareness work; absent on old rows
    prompt_sha256: Option<&'a str>,
    forge_role: Option<&'a str>,
    phase: &'a str,
}

#[derive(serde::Serialize)]
struct Envelope<'a, T> {
    schema_version: u32,
    kind: &'a str,
    timestamp_ms: i64,
    payload: T,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Read the export watermark (the last `request_id` already exported).
fn watermark(conn: &Connection) -> Result<String, HistDbError> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pamp_export_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        [],
    )?;
    let got: Option<String> = conn
        .query_row(
            "SELECT value FROM pamp_export_state WHERE key = ?1",
            [WATERMARK_KEY],
            |r| r.get(0),
        )
        .ok();
    Ok(got.unwrap_or_default())
}

fn set_watermark(conn: &Connection, v: &str) -> Result<(), HistDbError> {
    conn.execute(
        "INSERT INTO pamp_export_state (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [WATERMARK_KEY, v],
    )?;
    Ok(())
}

/// Export new scrape rows as PAMP `moe_route` records. Returns (requests, routes).
///
/// `limit_requests` bounds one pass so a first run over a large database does
/// not write a hundred-megabyte line-delimited file in one go.
pub fn export_routes(
    db: &str,
    out: &Path,
    limit_requests: usize,
) -> Result<(usize, usize), HistDbError> {
    let conn = Connection::open(db)?;
    let after = watermark(&conn)?;

    // Requests that have activations, in id order, past the watermark.
    let mut q = conn.prepare(
        "SELECT r.request_id, r.model_id, r.task_domain, r.prompt_sha256,
                r.forge_run_id, r.forge_role, r.created_at
         FROM requests r
         WHERE r.request_id > ?1
           AND EXISTS (SELECT 1 FROM activations a WHERE a.request_id = r.request_id)
         ORDER BY r.request_id
         LIMIT ?2",
    )?;
    let reqs: Vec<(String, String, Option<String>, Option<String>, Option<String>, Option<String>, String)> =
        q.query_map(rusqlite::params![after, limit_requests as i64], |row| {
            Ok((
                row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                row.get(4)?, row.get(5)?, row.get(6)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    if reqs.is_empty() {
        return Ok((0, 0));
    }

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(out)?;

    let mut n_routes = 0usize;
    let mut last_id = after;

    for (rid, model_id, task_domain, prompt_sha, forge_run, forge_role, created_at) in &reqs {
        // token_pos -> token_id, when the scrape carried the tokens
        let mut toks: BTreeMap<i64, i64> = BTreeMap::new();
        {
            let mut ts = conn.prepare(
                "SELECT pos, token_id FROM request_tokens WHERE request_id = ?1",
            )?;
            for row in ts.query_map([rid], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))? {
                let (p, t) = row?;
                toks.insert(p, t);
            }
        }

        // activations are one row per selected expert; regroup to one record
        // per (phase, layer, token_pos) with the experts in rank order.
        let mut a = conn.prepare(
            "SELECT phase, layer, token_pos, rank, expert_id
             FROM activations WHERE request_id = ?1
             ORDER BY phase, layer, token_pos, rank",
        )?;
        let rows: Vec<(String, i64, i64, i64, i64)> = a
            .query_map([rid], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let ts_ns = chrono_ns(created_at);
        let mut cur: Option<(String, i64, i64)> = None;
        let mut experts: Vec<i64> = Vec::new();

        let flush = |key: &(String, i64, i64), experts: &mut Vec<i64>, f: &mut std::fs::File|
            -> Result<(), HistDbError> {
            if experts.is_empty() {
                return Ok(());
            }
            let rec = Envelope {
                schema_version: 2,
                kind: "moe_route",
                timestamp_ms: now_ms(),
                payload: MoeRoute {
                    // forge_run_id when the caller supplied one, else the
                    // request id -- never a fabricated identifier.
                    run_id: forge_run.as_deref().unwrap_or(rid.as_str()),
                    model_id,
                    task_class: task_domain.as_deref(),
                    layer: key.1,
                    token_index: key.2,
                    expert_ids: std::mem::take(experts),
                    expert_weights: None,
                    expert_logits: None,
                    token_id: toks.get(&key.2).copied(),
                    timestamp_ns: ts_ns,
                    prompt_sha256: prompt_sha.as_deref(),
                    forge_role: forge_role.as_deref(),
                    phase: key.0.as_str(),
                },
            };
            writeln!(f, "{}", serde_json::to_string(&rec)?)?;
            Ok(())
        };

        for (phase, layer, tpos, _rank, eid) in rows {
            let key = (phase.clone(), layer, tpos);
            match &cur {
                Some(k) if *k == key => experts.push(eid),
                Some(k) => {
                    flush(k, &mut experts, &mut f)?;
                    n_routes += 1;
                    experts.push(eid);
                    cur = Some(key);
                }
                None => {
                    experts.push(eid);
                    cur = Some(key);
                }
            }
        }
        if let Some(k) = &cur {
            flush(k, &mut experts, &mut f)?;
            n_routes += 1;
        }

        last_id = rid.clone();
    }

    f.flush()?;
    set_watermark(&conn, &last_id)?;
    Ok((reqs.len(), n_routes))
}

/// `created_at` is stored as `YYYY-MM-DD HH:MM:SS` (sqlite `datetime('now')`).
/// Parsed without a date dependency; falls back to now on anything unexpected
/// rather than emitting a zero that would look like 1970.
fn chrono_ns(s: &str) -> i64 {
    fn p(x: &str) -> Option<i64> {
        let (d, t) = x.split_once(' ')?;
        let mut di = d.split('-');
        let (y, mo, da) = (
            di.next()?.parse::<i64>().ok()?,
            di.next()?.parse::<i64>().ok()?,
            di.next()?.parse::<i64>().ok()?,
        );
        let mut ti = t.split(':');
        let (h, mi, se) = (
            ti.next()?.parse::<i64>().ok()?,
            ti.next()?.parse::<i64>().ok()?,
            ti.next()?.split('.').next()?.parse::<i64>().ok()?,
        );
        // days from civil (Howard Hinnant's algorithm)
        let y2 = if mo <= 2 { y - 1 } else { y };
        let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
        let yoe = y2 - era * 400;
        let mp = (mo + 9) % 12;
        let doy = (153 * mp + 2) / 5 + da - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146097 + doe - 719468;
        Some(((days * 86400) + h * 3600 + mi * 60 + se) * 1_000_000_000)
    }
    p(s).unwrap_or_else(|| now_ms() * 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_date_to_ns_is_right() {
        // 2026-11-04 00:00:00 UTC = 1793750400 s
        assert_eq!(chrono_ns("2026-11-04 00:00:00"), 1_793_750_400 * 1_000_000_000);
        // epoch itself
        assert_eq!(chrono_ns("1970-01-01 00:00:00"), 0);
        // a known leap-year boundary
        assert_eq!(chrono_ns("2024-02-29 12:00:00"), 1_709_208_000 * 1_000_000_000);
    }

    #[test]
    fn unparseable_timestamp_does_not_become_1970() {
        // A zero here would silently date every row to the epoch and poison
        // any time-ordered analysis downstream.
        let got = chrono_ns("not a timestamp");
        assert!(got > 1_700_000_000_000_000_000, "fell back to {got}");
    }
}
