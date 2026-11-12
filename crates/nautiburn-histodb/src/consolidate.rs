//! The consolidated store: rules, co-firing, per-request summaries, and the one-off conversion of a
//! per-activation DB into it (`nautiburn-histodb consolidate`).
//!
//! Keep policy (shared by the conversion and by every later ingest): a request's activation rows
//! stay in the DB only when the frozen rule for its (model, class, phase) served it worse than
//! usual, when there is no rule yet, when it is a multi-token decode run, or when it falls in the
//! random sample. Everything else is counted into `histogram_counts` / `cofire_counts`, gets a
//! `request_summary` row, and its rows are not stored.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

use crate::HistDbError;

/// Tunables, read from `meta` (written there by migration 007).
#[derive(Debug, Clone, Copy)]
pub struct KeepPolicy {
    /// keep rows when the hit rate is this far below the rule's median
    pub margin: f64,
    /// keep 1 request in this many regardless (0 = no sample)
    pub sample_every: u64,
    /// keep multi-token decode runs
    pub decode: bool,
}

impl KeepPolicy {
    pub fn load(conn: &Connection) -> Result<Self, HistDbError> {
        let get = |k: &str| -> Result<Option<String>, HistDbError> {
            Ok(conn
                .query_row("SELECT value FROM meta WHERE key = ?1", params![k], |r| r.get(0))
                .optional()?)
        };
        Ok(Self {
            margin: get("keep_margin")?.and_then(|v| v.parse().ok()).unwrap_or(0.05),
            sample_every: get("keep_sample_every")?.and_then(|v| v.parse().ok()).unwrap_or(50),
            decode: get("keep_decode")?.map(|v| v != "0").unwrap_or(true),
        })
    }

    /// Why this request's rows are kept, or `None` to store only its aggregates.
    /// `scored`: (hits, selections, the rule's median hit rate) when a rule exists.
    pub fn reason(
        &self,
        request_id: &str,
        phase: &str,
        n_tokens: i64,
        scored: Option<(i64, i64, f64)>,
    ) -> Option<&'static str> {
        let Some((hits, selections, median)) = scored else {
            return Some("no_rule");
        };
        if selections > 0 && (hits as f64 / selections as f64) < median - self.margin {
            return Some("diverged");
        }
        if self.decode && phase == "decode" && n_tokens > 1 {
            return Some("decode");
        }
        if self.sample_every > 0 && fnv1a(request_id) % self.sample_every == 0 {
            return Some("sample");
        }
        None
    }
}

/// Stable across runs and builds, so the sample is the same requests every time.
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The newest rule of one (model, class, phase).
#[derive(Debug, Clone)]
pub struct Rule {
    pub version: String,
    pub median_hit: f64,
    pub set: HashSet<(i64, i64)>,
}

pub fn load_rule(
    conn: &Connection,
    model_id: &str,
    class_id: &str,
    phase: &str,
) -> Result<Option<Rule>, HistDbError> {
    let head: Option<(String, f64)> = conn
        .query_row(
            "SELECT version, median_hit FROM predict_rules
             WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3
             ORDER BY built_at DESC, version DESC LIMIT 1",
            params![model_id, class_id, phase],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((version, median_hit)) = head else {
        return Ok(None);
    };
    let mut stmt = conn.prepare(
        "SELECT layer, expert_id FROM rule_experts
         WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3 AND version = ?4",
    )?;
    let set = stmt
        .query_map(params![model_id, class_id, phase, version], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<HashSet<(i64, i64)>, _>>()?;
    Ok(Some(Rule { version, median_hit, set }))
}

/// Top `keep_frac` experts of every layer by count (ties: lower expert id first).
pub fn pick_set(cells: &BTreeMap<(i64, i64), i64>, n_experts: i64, keep_frac: f64) -> HashSet<(i64, i64)> {
    let keep = ((n_experts as f64 * keep_frac).floor() as usize).max(1);
    let mut by_layer: BTreeMap<i64, Vec<(i64, i64)>> = BTreeMap::new();
    for (&(layer, expert), &count) in cells {
        by_layer.entry(layer).or_default().push((count, expert));
    }
    let mut set = HashSet::new();
    for (layer, mut v) in by_layer {
        v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, expert) in v.into_iter().take(keep) {
            set.insert((layer, expert));
        }
    }
    set
}

/// Co-firing pairs of one request: rows are `(token_pos, layer, experts)`; each token's layers are
/// paired with that token's next layer present. Returns the pair counts and how many pairs.
pub fn cofire_pairs(rows: &mut Vec<(i64, i64, Vec<i64>)>) -> (HashMap<(i64, i64, i64), i64>, i64) {
    rows.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut out: HashMap<(i64, i64, i64), i64> = HashMap::new();
    let mut total = 0i64;
    for w in rows.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.0 != b.0 || a.1 == b.1 {
            continue;
        }
        for &ea in &a.2 {
            for &eb in &b.2 {
                *out.entry((a.1, ea, eb)).or_insert(0) += 1;
                total += 1;
            }
        }
    }
    (out, total)
}

pub fn cofire_group_id(conn: &Connection, model_id: &str, class_id: &str, phase: &str) -> Result<i64, HistDbError> {
    conn.execute(
        "INSERT OR IGNORE INTO cofire_groups (model_id, class_id, phase) VALUES (?1, ?2, ?3)",
        params![model_id, class_id, phase],
    )?;
    Ok(conn.query_row(
        "SELECT group_id FROM cofire_groups WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3",
        params![model_id, class_id, phase],
        |r| r.get(0),
    )?)
}

/// What the conversion did, and whether the new DB accounts for the old one.
#[derive(Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    pub failures: Vec<String>,
}

impl Report {
    fn say(&mut self, s: String) {
        println!("{s}");
        self.lines.push(s);
    }
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        let line = format!("{} {name}: {detail}", if ok { "ok  " } else { "FAIL" });
        println!("{line}");
        if !ok {
            self.failures.push(line.clone());
        }
        self.lines.push(line);
    }
}

#[derive(Default)]
struct Group {
    model_id: String,
    class_id: String,
    phase: String,
    n_experts: i64,
    set: HashSet<(i64, i64)>,
    hist_selections: i64,
    cofire: HashMap<i64, Vec<u64>>, // layer -> [n_experts * n_experts]
    cofire_total: u64,
    scanned_selections: i64,
    hit_rates: Vec<f64>,
}

struct ReqOut {
    request_id: String,
    group: usize,
    n_tokens: i64,
    selections: i64,
    hits: i64,
    rows: i64,
}

/// Build `new_path` from `old_path` (opened read-only, never written).
///
/// `min_selections` / `min_requests`: evidence a (model, class, phase) needs before it gets a rule.
pub fn consolidate(
    old_path: &Path,
    new_path: &Path,
    keep_frac: f64,
    min_selections: i64,
    min_requests: usize,
    version: &str,
) -> Result<Report, HistDbError> {
    let mut rep = Report::default();
    if new_path.exists() {
        return Err(HistDbError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} exists; consolidate writes a new file", new_path.display()),
        )));
    }
    let old = Connection::open_with_flags(old_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    old.pragma_update(None, "cache_size", -4_194_304i64)?;
    drop(crate::HistDb::open(new_path)?); // creates the file at the current schema
    let new = Connection::open(new_path)?;
    new.pragma_update(None, "journal_mode", "WAL")?;
    new.pragma_update(None, "synchronous", "NORMAL")?;
    new.pragma_update(None, "foreign_keys", "OFF")?;
    new.pragma_update(None, "cache_size", -2_097_152i64)?;
    let policy = KeepPolicy::load(&new)?;

    // ── 1. every table except the per-activation ones, as is ────────────────────────────────
    let old_uri = format!("file:{}?mode=ro", old_path.display());
    new.execute("ATTACH DATABASE ?1 AS old", params![old_uri])?;
    let tables: Vec<(String, String)> = {
        let mut st = old.prepare(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             AND name NOT IN ('activations', 'activation_near') ORDER BY name",
        )?;
        let v = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        v
    };
    for (name, sql) in &tables {
        let exists: bool = new
            .query_row("SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = ?1", params![name], |_| Ok(true))
            .optional()?
            .unwrap_or(false);
        if !exists {
            new.execute_batch(sql)?;
        }
        let cols = |c: &Connection, schema: &str| -> Result<Vec<String>, HistDbError> {
            let mut st = c.prepare(&format!("PRAGMA {schema}.table_info('{name}')"))?;
            let v = st.query_map([], |r| r.get::<_, String>(1))?.collect::<Result<_, _>>()?;
            Ok(v)
        };
        let new_cols = cols(&new, "main")?;
        let shared: Vec<String> = cols(&new, "old")?.into_iter().filter(|c| new_cols.contains(c)).collect();
        let list = shared.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ");
        // meta holds the schema version and the keep policy of the NEW file: old keys never overwrite
        let verb = if name == "meta" { "INSERT OR IGNORE" } else { "INSERT OR REPLACE" };
        new.execute_batch(&format!(
            "BEGIN; {verb} INTO main.\"{name}\" ({list}) SELECT {list} FROM old.\"{name}\"; COMMIT;"
        ))?;
        let n_old: i64 = old.query_row(&format!("SELECT COUNT(*) FROM \"{name}\""), [], |r| r.get(0))?;
        let n_new: i64 = new.query_row(&format!("SELECT COUNT(*) FROM main.\"{name}\""), [], |r| r.get(0))?;
        let ok = if name == "meta" { n_new >= n_old } else { n_new == n_old };
        rep.check(&format!("table {name}"), ok, format!("old {n_old} rows, new {n_new}"));
    }
    let hist = |c: &Connection, t: &str| -> Result<(i64, i64), HistDbError> {
        Ok(c.query_row(
            &format!("SELECT COALESCE(SUM(count), 0), COALESCE(SUM(count * (layer + 1) * (expert_id + 7)), 0) FROM {t}"),
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?)
    };
    let (ho, hn) = (hist(&old, "histogram_counts")?, hist(&new, "main.histogram_counts")?);
    rep.check("histogram mass and checksum", ho == hn, format!("old {ho:?}, new {hn:?}"));

    // ── 2. groups, and a candidate set per group from the histogram ─────────────────────────
    let n_experts: HashMap<String, i64> = {
        let mut st = old.prepare("SELECT model_id, n_experts FROM models")?;
        let v = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        v
    };
    let mut groups: Vec<Group> = Vec::new();
    let mut group_of: HashMap<(String, String, String), usize> = HashMap::new();
    {
        let mut cells: BTreeMap<(String, String, String), BTreeMap<(i64, i64), i64>> = BTreeMap::new();
        let mut st = old.prepare("SELECT model_id, class_id, phase, layer, expert_id, count FROM histogram_counts")?;
        let mut rows = st.query([])?;
        while let Some(r) = rows.next()? {
            cells
                .entry((r.get(0)?, r.get(1)?, r.get(2)?))
                .or_default()
                .insert((r.get(3)?, r.get(4)?), r.get(5)?);
        }
        for (key, c) in cells {
            let ne = *n_experts.get(&key.0).unwrap_or(&0);
            group_of.insert(key.clone(), groups.len());
            groups.push(Group {
                model_id: key.0,
                class_id: key.1,
                phase: key.2,
                n_experts: ne,
                set: pick_set(&c, ne, keep_frac),
                hist_selections: c.values().sum(),
                ..Default::default()
            });
        }
    }
    let req_info: HashMap<String, (String, String, i64)> = {
        let mut st = old.prepare("SELECT request_id, model_id, class_id, COALESCE(n_tokens, 0) FROM requests")?;
        let v = st
            .query_map([], |r| Ok((r.get::<_, String>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?))))?
            .collect::<Result<_, _>>()?;
        v
    };
    rep.say(format!("groups {} requests {}", groups.len(), req_info.len()));

    // ── 3. one pass over the activations in storage order ───────────────────────────────────
    let mut outs: Vec<ReqOut> = Vec::new();
    let mut scanned_rows: i64 = 0;
    let mut orphan_rows: i64 = 0;
    {
        let mut cur_id = String::new();
        let mut cur_phase = String::new();
        let mut buf: Vec<(i64, i64, i64, i64)> = Vec::new(); // token_pos, layer, rank, expert
        let mut finish = |id: &str, phase: &str, buf: &mut Vec<(i64, i64, i64, i64)>, groups: &mut Vec<Group>,
                          group_of: &mut HashMap<(String, String, String), usize>| {
            if buf.is_empty() {
                return;
            }
            let Some((model, class, n_tokens)) = req_info.get(id) else {
                orphan_rows += buf.len() as i64;
                buf.clear();
                return;
            };
            let key = (model.clone(), class.clone(), phase.to_string());
            let gi = *group_of.entry(key).or_insert_with(|| {
                groups.push(Group {
                    model_id: model.clone(),
                    class_id: class.clone(),
                    phase: phase.to_string(),
                    n_experts: *n_experts.get(model).unwrap_or(&0),
                    ..Default::default()
                });
                groups.len() - 1
            });
            let g = &mut groups[gi];
            buf.sort_unstable();
            let ne = g.n_experts.max(1) as usize;
            let mut hits = 0i64;
            // walk (token, layer) runs; pair each with the same token's next layer
            let mut i = 0usize;
            let mut prev: Option<(i64, i64, usize, usize)> = None; // token, layer, start, end
            while i < buf.len() {
                let (tok, layer) = (buf[i].0, buf[i].1);
                let mut j = i;
                while j < buf.len() && buf[j].0 == tok && buf[j].1 == layer {
                    if g.set.contains(&(layer, buf[j].3)) {
                        hits += 1;
                    }
                    j += 1;
                }
                if let Some((ptok, player, ps, pe)) = prev {
                    if ptok == tok {
                        let m = g.cofire.entry(player).or_insert_with(|| vec![0u64; ne * ne]);
                        for a in ps..pe {
                            for b in i..j {
                                let (ea, eb) = (buf[a].3 as usize, buf[b].3 as usize);
                                if ea < ne && eb < ne {
                                    m[ea * ne + eb] += 1;
                                    g.cofire_total += 1;
                                }
                            }
                        }
                    }
                }
                prev = Some((tok, layer, i, j));
                i = j;
            }
            let selections = buf.len() as i64;
            g.scanned_selections += selections;
            if !g.set.is_empty() && selections > 0 {
                g.hit_rates.push(hits as f64 / selections as f64);
            }
            outs.push(ReqOut { request_id: id.to_string(), group: gi, n_tokens: *n_tokens, selections, hits, rows: selections });
            buf.clear();
        };
        let mut st = old.prepare("SELECT request_id, phase, token_pos, layer, rank, expert_id FROM activations ORDER BY rowid")?;
        let mut rows = st.query([])?;
        while let Some(r) = rows.next()? {
            let id = r.get_ref(0)?.as_str().map_err(rusqlite::Error::from)?;
            let phase = r.get_ref(1)?.as_str().map_err(rusqlite::Error::from)?;
            if id != cur_id || phase != cur_phase {
                finish(&cur_id, &cur_phase, &mut buf, &mut groups, &mut group_of);
                cur_id.clear();
                cur_id.push_str(id);
                cur_phase.clear();
                cur_phase.push_str(phase);
            }
            buf.push((r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?));
            scanned_rows += 1;
            if scanned_rows % 100_000_000 == 0 {
                println!("scanned {scanned_rows} activation rows");
            }
        }
        finish(&cur_id, &cur_phase, &mut buf, &mut groups, &mut group_of);
    }
    rep.say(format!("scanned {scanned_rows} activation rows in {} request runs ({orphan_rows} rows had no request)", outs.len()));

    // ── 4. rules ───────────────────────────────────────────────────────────────────────────
    let mut medians: Vec<Option<f64>> = Vec::with_capacity(groups.len());
    new.execute_batch("BEGIN")?;
    for g in &mut groups {
        g.hit_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let enough = g.hist_selections >= min_selections && g.hit_rates.len() >= min_requests && !g.set.is_empty();
        if !enough {
            medians.push(None);
            continue;
        }
        let median = g.hit_rates[g.hit_rates.len() / 2];
        medians.push(Some(median));
        new.execute(
            "INSERT INTO predict_rules (model_id, class_id, phase, version, keep_frac, selections, requests, median_hit)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![g.model_id, g.class_id, g.phase, version, keep_frac, g.hist_selections, g.hit_rates.len() as i64, median],
        )?;
        let mut st = new.prepare_cached(
            "INSERT INTO rule_experts (model_id, class_id, phase, version, layer, expert_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for &(layer, expert) in &g.set {
            st.execute(params![g.model_id, g.class_id, g.phase, version, layer, expert])?;
        }
    }
    new.execute_batch("COMMIT")?;
    for (g, m) in groups.iter().zip(&medians) {
        rep.say(format!(
            "group {} | {} | {}: histogram {} selections, rows scanned {} ({:.1}%), requests scored {}, rule {}",
            g.model_id, g.class_id, g.phase, g.hist_selections, g.scanned_selections,
            if g.hist_selections > 0 { 100.0 * g.scanned_selections as f64 / g.hist_selections as f64 } else { 0.0 },
            g.hit_rates.len(),
            m.map(|v| format!("{version} median hit {v:.3}")).unwrap_or_else(|| "none (too little evidence)".into()),
        ));
    }

    // ── 5. co-firing ───────────────────────────────────────────────────────────────────────
    let mut cofire_written: u64 = 0;
    let mut cofire_expected: u64 = 0;
    new.execute_batch("BEGIN")?;
    for g in &groups {
        if g.cofire.is_empty() {
            continue;
        }
        cofire_expected += g.cofire_total;
        let gid = cofire_group_id(&new, &g.model_id, &g.class_id, &g.phase)?;
        let ne = g.n_experts.max(1) as usize;
        let mut st = new.prepare_cached(
            "INSERT INTO cofire_counts (group_id, layer, expert_a, expert_b, count) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        let mut layers: Vec<_> = g.cofire.keys().copied().collect();
        layers.sort_unstable();
        for layer in layers {
            for (i, &n) in g.cofire[&layer].iter().enumerate() {
                if n > 0 {
                    st.execute(params![gid, layer, (i / ne) as i64, (i % ne) as i64, n as i64])?;
                    cofire_written += n;
                }
            }
        }
    }
    new.execute_batch("COMMIT")?;
    let cofire_db: i64 = new.query_row("SELECT COALESCE(SUM(count), 0) FROM cofire_counts", [], |r| r.get(0))?;
    rep.check(
        "co-firing pairs",
        cofire_db as u64 == cofire_expected && cofire_written == cofire_expected,
        format!("counted {cofire_expected}, stored {cofire_db}"),
    );

    // ── 6. summaries and the keep list ─────────────────────────────────────────────────────
    // a request can come back as more than one run (rows of two files interleaved): merge first
    let mut merged: BTreeMap<(String, usize), ReqOut> = BTreeMap::new();
    for o in outs {
        match merged.get_mut(&(o.request_id.clone(), o.group)) {
            Some(m) => {
                m.selections += o.selections;
                m.hits += o.hits;
                m.rows += o.rows;
            }
            None => {
                merged.insert((o.request_id.clone(), o.group), o);
            }
        }
    }
    new.execute_batch("BEGIN; CREATE TEMP TABLE keep_ids (request_id TEXT PRIMARY KEY) WITHOUT ROWID;")?;
    let mut kept_rows_expected: i64 = 0;
    let mut kept: BTreeMap<&'static str, (i64, i64)> = BTreeMap::new();
    let mut dropped = (0i64, 0i64);
    let mut summary_ids: HashSet<String> = HashSet::new();
    for ((id, gi), o) in &merged {
        let g = &groups[*gi];
        let scored = medians[*gi].map(|m| (o.hits, o.selections, m));
        let reason = policy.reason(id, &g.phase, o.n_tokens, scored);
        if let Some(why) = reason {
            let e = kept.entry(why).or_insert((0, 0));
            e.0 += 1;
            e.1 += o.rows;
            kept_rows_expected += o.rows;
            new.execute("INSERT OR IGNORE INTO keep_ids (request_id) VALUES (?1)", params![id])?;
        } else {
            dropped.0 += 1;
            dropped.1 += o.rows;
        }
        // one summary per request: a request with rows in two phases keeps the first and is always kept
        if summary_ids.insert(id.clone()) {
            new.execute(
                "INSERT INTO request_summary (request_id, model_id, class_id, phase, n_tokens, selections, hits, rule_version, rows_kept, reason)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![id, g.model_id, g.class_id, g.phase, o.n_tokens, o.selections,
                        scored.map(|s| s.0), scored.map(|_| version), reason.is_some() as i64, reason],
            )?;
        } else {
            new.execute("INSERT OR IGNORE INTO keep_ids (request_id) VALUES (?1)", params![id])?;
            new.execute("UPDATE request_summary SET rows_kept = 1, reason = COALESCE(reason, 'two_phases') WHERE request_id = ?1", params![id])?;
        }
    }
    new.execute_batch("COMMIT")?;
    for (why, (n, rows)) in &kept {
        rep.say(format!("kept {why}: {n} requests, {rows} rows"));
    }
    rep.say(format!("not kept (aggregates only): {} requests, {} rows", dropped.0, dropped.1));
    let n_summary: i64 = new.query_row("SELECT COUNT(*) FROM request_summary", [], |r| r.get(0))?;
    rep.check("request summaries", n_summary as usize == summary_ids.len(), format!("{n_summary} rows for {} requests with activations", summary_ids.len()));

    // ── 7. copy the kept rows ──────────────────────────────────────────────────────────────
    let copy = |table: &str| -> Result<(i64, i64), HistDbError> {
        let cols = {
            let mut st = new.prepare(&format!("PRAGMA old.table_info('{table}')"))?;
            let v: Vec<String> = st.query_map([], |r| r.get::<_, String>(1))?.collect::<Result<_, _>>()?;
            v.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ")
        };
        new.execute_batch(&format!(
            "BEGIN; INSERT INTO main.{table} ({cols}) SELECT {cols} FROM old.{table}
             WHERE request_id IN (SELECT request_id FROM keep_ids); COMMIT;"
        ))?;
        let n_new: i64 = new.query_row(&format!("SELECT COUNT(*) FROM main.{table}"), [], |r| r.get(0))?;
        let n_old: i64 = new.query_row(
            &format!("SELECT COUNT(*) FROM old.{table} WHERE request_id IN (SELECT request_id FROM keep_ids)"),
            [],
            |r| r.get(0),
        )?;
        Ok((n_old, n_new))
    };
    let (ao, an) = copy("activations")?;
    // a kept request's rows, counted in the pass, must all be there (a request kept for one phase
    // brings its other phase along, so the copy may hold more than the pass counted as kept)
    rep.check("kept activation rows", ao == an && an >= kept_rows_expected, format!("old {ao}, new {an}, counted in the pass {kept_rows_expected}"));
    let has_near: bool = old
        .query_row("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'activation_near'", [], |_| Ok(true))
        .optional()?
        .unwrap_or(false);
    if has_near {
        let (no, nn) = copy("activation_near")?;
        rep.check("kept runner-up rows", no == nn, format!("old {no}, new {nn}"));
    }
    let total_old: i64 = old.query_row("SELECT COUNT(*) FROM activations", [], |r| r.get(0))?;
    rep.check("every activation row was read", total_old == scanned_rows, format!("table holds {total_old}, pass read {scanned_rows}"));

    new.execute_batch("DETACH DATABASE old; PRAGMA wal_checkpoint(TRUNCATE);")?;
    let quick: String = new.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    rep.check("new file quick_check", quick == "ok", quick.clone());
    let fk: i64 = new.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| r.get(0))?;
    rep.check("foreign keys", fk == 0, format!("{fk} violations"));
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_join_a_tokens_layer_with_its_next_layer_only() {
        let mut rows = vec![(0, 1, vec![5, 6]), (0, 0, vec![1, 2]), (1, 0, vec![3, 4]), (0, 3, vec![7, 8])];
        let (pairs, total) = cofire_pairs(&mut rows);
        // token 0: layer 0 -> 1 and 1 -> 3; token 1 has one layer, so no pair
        assert_eq!(total, 8);
        assert_eq!(pairs[&(0, 1, 5)], 1);
        assert_eq!(pairs[&(1, 6, 8)], 1);
        assert!(!pairs.contains_key(&(0, 3, 5)));
    }

    #[test]
    fn keep_reasons_in_order() {
        let p = KeepPolicy { margin: 0.05, sample_every: 0, decode: true };
        assert_eq!(p.reason("r", "prefill", 100, None), Some("no_rule"));
        assert_eq!(p.reason("r", "prefill", 100, Some((80, 100, 0.90))), Some("diverged"));
        assert_eq!(p.reason("r", "prefill", 100, Some((88, 100, 0.90))), None);
        assert_eq!(p.reason("r", "decode", 100, Some((88, 100, 0.90))), Some("decode"));
        assert_eq!(p.reason("r", "decode", 1, Some((88, 100, 0.90))), None);
    }

    #[test]
    fn the_set_is_the_top_share_of_each_layer() {
        let mut c = BTreeMap::new();
        for e in 0..8 {
            c.insert((0, e), e);
            c.insert((1, e), 8 - e);
        }
        let s = pick_set(&c, 8, 0.25);
        assert_eq!(s.len(), 4);
        assert!(s.contains(&(0, 7)) && s.contains(&(0, 6)) && s.contains(&(1, 0)) && s.contains(&(1, 1)));
    }
}
