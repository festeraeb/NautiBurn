//! The run envelope: correlate a prompt with the experts it routed to, at the
//! time the work happens, rather than reconstructing it afterwards.
//!
//! Lemonade sees the prompt (it is the front door every request passes
//! through). Only the engine can see expert selection (it happens in the
//! kernels). Neither can produce the join alone, and a timestamp-window
//! correlation is lossy under concurrency and fails *silently* — it returns a
//! wrong pairing rather than no pairing.
//!
//! So Lemonade mints a `run_id` where the prompt is visible, and every card
//! that processes part of that run appends a hop carrying its own scrape
//! reference. The chain is the correlation.
//!
//! # Retention
//!
//! Three classes, and the third is the one that is easy to leave out and
//! expensive to omit:
//!
//! - `divergence` — only what differed from PAMP's prediction. At the fleet's
//!   measured 3.39/8 overlap this saves ~40%, improving as the predictor does.
//! - `full_lowscore` — the prediction was bad, so keep everything; this is the
//!   material worth analysing.
//! - `full_control` — kept regardless of score, **sampled before the score is
//!   known**. Without it the archive of full records is a library of the
//!   predictor's failures: training on it teaches the model its own mistakes,
//!   and "hit-rate improved" cannot be falsified because the baseline is
//!   selected by the thing being measured.
//!
//! A divergence row without `predictor_version` is rejected. A delta cannot be
//! reconstructed without the predictor that produced it, and a retrain would
//! silently render it unreadable.

use rusqlite::{params, Connection};

use crate::schema::HistDbError;

/// How much of a hop's scrape was kept, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// Only the experts that differed from prediction.
    Divergence,
    /// Prediction was poor; the whole record was kept for analysis.
    FullLowScore,
    /// Kept regardless of score, sampled *before* scoring. The honest baseline.
    FullControl,
}

impl Retention {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Divergence => "divergence",
            Self::FullLowScore => "full_lowscore",
            Self::FullControl => "full_control",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "full_lowscore" => Self::FullLowScore,
            "full_control" => Self::FullControl,
            _ => Self::Divergence,
        }
    }

    /// Whether this class holds a complete record (so it can be re-analysed
    /// without the predictor).
    pub fn is_full(&self) -> bool {
        matches!(self, Self::FullLowScore | Self::FullControl)
    }
}

/// Decide retention for a hop.
///
/// `control_roll` is a uniform [0,1) draw taken **before** the score is known —
/// passing the score in and sampling afterwards would bias the control set and
/// defeat its purpose. `score` is `None` when PAMP had no prediction to score
/// against, which is itself a reason to keep the record whole.
pub fn classify(score: Option<f64>, cutoff: f64, control_rate: f64, control_roll: f64) -> Retention {
    if control_roll < control_rate {
        return Retention::FullControl;
    }
    match score {
        None => Retention::FullLowScore, // nothing predicted it; keep it all
        Some(s) if s < cutoff => Retention::FullLowScore,
        Some(_) => Retention::Divergence,
    }
}

/// One card's contribution to a run.
#[derive(Debug, Clone)]
pub struct Hop {
    pub card: Option<String>,
    pub engine: Option<String>,
    /// The engine's own request id, joining to `activations`.
    pub request_id: Option<String>,
    pub scrape_ref: Option<String>,
    pub scrape_rows: Option<i64>,
    /// Digest of the scrape, so a later read can be proven to be this one.
    pub scrape_sha256: Option<String>,
    pub t_start_ms: Option<i64>,
    pub t_end_ms: Option<i64>,
    pub score: Option<f64>,
    pub predictor_version: Option<String>,
    pub retention: Retention,
}

impl Hop {
    pub fn new(card: impl Into<String>, engine: impl Into<String>) -> Self {
        Self {
            card: Some(card.into()),
            engine: Some(engine.into()),
            request_id: None,
            scrape_ref: None,
            scrape_rows: None,
            scrape_sha256: None,
            t_start_ms: None,
            t_end_ms: None,
            score: None,
            predictor_version: None,
            retention: Retention::Divergence,
        }
    }

    pub fn with_scrape(
        mut self,
        request_id: impl Into<String>,
        scrape_ref: impl Into<String>,
        rows: i64,
        sha256: impl Into<String>,
    ) -> Self {
        self.request_id = Some(request_id.into());
        self.scrape_ref = Some(scrape_ref.into());
        self.scrape_rows = Some(rows);
        self.scrape_sha256 = Some(sha256.into());
        self
    }

    pub fn scored(mut self, score: f64, predictor_version: impl Into<String>) -> Self {
        self.score = Some(score);
        self.predictor_version = Some(predictor_version.into());
        self
    }
}

/// Open an envelope. `run_id` comes from Lemonade; this never mints one,
/// because an id invented downstream cannot be traced back to the request.
pub fn begin_run(
    conn: &Connection,
    run_id: &str,
    prompt_sha256: Option<&str>,
    prompt_text: Option<&str>,
    task_class: Option<&str>,
    forge_role: Option<&str>,
    model_id: Option<&str>,
) -> Result<(), HistDbError> {
    if run_id.trim().is_empty() {
        return Err(HistDbError::InvalidData("run_id must not be empty".into()));
    }
    conn.execute(
        "INSERT INTO run_envelope
           (run_id, prompt_sha256, prompt_text, task_class, forge_role, model_id, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'open')
         ON CONFLICT(run_id) DO UPDATE SET
           prompt_sha256 = COALESCE(excluded.prompt_sha256, run_envelope.prompt_sha256),
           prompt_text   = COALESCE(excluded.prompt_text,   run_envelope.prompt_text),
           task_class    = COALESCE(excluded.task_class,    run_envelope.task_class),
           forge_role    = COALESCE(excluded.forge_role,    run_envelope.forge_role),
           model_id      = COALESCE(excluded.model_id,      run_envelope.model_id)",
        params![run_id, prompt_sha256, prompt_text, task_class, forge_role, model_id],
    )?;
    Ok(())
}

/// Append a hop. Returns its sequence number.
///
/// Appends only: an existing hop is never rewritten, so a card that died
/// leaves a gap in the sequence instead of an envelope that looks complete.
pub fn append_hop(conn: &Connection, run_id: &str, hop: &Hop) -> Result<i64, HistDbError> {
    if hop.retention == Retention::Divergence && hop.predictor_version.is_none() {
        return Err(HistDbError::InvalidData(
            "a divergence hop needs predictor_version: the delta cannot be \
             reconstructed without the predictor that produced it"
                .into(),
        ));
    }
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM run_envelope WHERE run_id = ?1",
        [run_id],
        |r| r.get(0),
    )?;
    if exists == 0 {
        return Err(HistDbError::InvalidData(format!(
            "unknown run_id {run_id}; begin_run first so the prompt is recorded"
        )));
    }
    let seq: i64 = conn.query_row(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_hop WHERE run_id = ?1",
        [run_id],
        |r| r.get(0),
    )?;
    conn.execute(
        "INSERT INTO run_hop
           (run_id, seq, card, engine, request_id, scrape_ref, scrape_rows,
            scrape_sha256, t_start_ms, t_end_ms, score, predictor_version, retention)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
        params![
            run_id, seq, hop.card, hop.engine, hop.request_id, hop.scrape_ref,
            hop.scrape_rows, hop.scrape_sha256, hop.t_start_ms, hop.t_end_ms,
            hop.score, hop.predictor_version, hop.retention.as_str()
        ],
    )?;
    Ok(seq)
}

/// Close an envelope.
pub fn complete_run(conn: &Connection, run_id: &str) -> Result<(), HistDbError> {
    conn.execute(
        "UPDATE run_envelope SET status = 'complete', completed_at = datetime('now')
         WHERE run_id = ?1",
        [run_id],
    )?;
    Ok(())
}

/// Envelopes still open past `older_than_secs` — a card took the work and
/// never came back. Reported, never auto-closed: a silent close would hide
/// exactly the failure this is here to surface.
pub fn stale_runs(conn: &Connection, older_than_secs: i64) -> Result<Vec<String>, HistDbError> {
    let mut q = conn.prepare(
        "SELECT run_id FROM run_envelope
         WHERE status = 'open'
           AND created_at < datetime('now', ?1)
         ORDER BY created_at",
    )?;
    let arg = format!("-{older_than_secs} seconds");
    let rows = q.query_map([arg], |r| r.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HistDb;

    fn db() -> Connection {
        let d = HistDb::open_in_memory().expect("open");
        drop(d);
        // open_in_memory gives a fresh db each call, so build one directly
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(crate::schema::MIGRATION_1).ok();
        c.execute_batch(crate::schema::MIGRATION_5).unwrap();
        c
    }

    #[test]
    fn control_is_sampled_before_scoring() {
        // A roll under the rate wins regardless of score -- including a score
        // that would otherwise be "good". That is the point: the control set
        // must not be selected by the thing it is controlling for.
        assert_eq!(classify(Some(8.0), 4.0, 0.02, 0.001), Retention::FullControl);
        assert_eq!(classify(Some(0.0), 4.0, 0.02, 0.001), Retention::FullControl);
    }

    #[test]
    fn low_score_keeps_the_whole_record() {
        assert_eq!(classify(Some(1.0), 4.0, 0.02, 0.5), Retention::FullLowScore);
    }

    #[test]
    fn good_score_keeps_only_the_divergence() {
        assert_eq!(classify(Some(7.0), 4.0, 0.02, 0.5), Retention::Divergence);
    }

    #[test]
    fn no_prediction_means_keep_everything() {
        // Nothing predicted it, so there is no delta to express it against.
        assert_eq!(classify(None, 4.0, 0.02, 0.5), Retention::FullLowScore);
        assert!(classify(None, 4.0, 0.02, 0.5).is_full());
    }

    #[test]
    fn divergence_without_predictor_version_is_rejected() {
        let c = db();
        begin_run(&c, "run-1", Some("abc"), None, Some("rust"), None, Some("m")).unwrap();
        let hop = Hop::new("cesarops4om:1", "llama.cpp-nautiburn");
        let err = append_hop(&c, "run-1", &hop).unwrap_err();
        assert!(err.to_string().contains("predictor_version"), "{err}");
    }

    #[test]
    fn hops_append_and_sequence() {
        let c = db();
        begin_run(&c, "run-2", None, None, None, None, None).unwrap();
        let h1 = Hop::new("c4:1", "llama.cpp").scored(6.0, "pamp-v3");
        let h2 = Hop::new("c4:3", "llama.cpp").scored(5.0, "pamp-v3");
        assert_eq!(append_hop(&c, "run-2", &h1).unwrap(), 1);
        assert_eq!(append_hop(&c, "run-2", &h2).unwrap(), 2);
    }

    #[test]
    fn a_hop_for_an_unknown_run_is_refused() {
        // Otherwise a hop could exist with no prompt behind it, which is the
        // un-joinable state this whole design removes.
        let c = db();
        let hop = Hop::new("c4:1", "llama.cpp").scored(6.0, "pamp-v3");
        let err = append_hop(&c, "never-opened", &hop).unwrap_err();
        assert!(err.to_string().contains("unknown run_id"), "{err}");
    }

    #[test]
    fn scrape_is_referenced_not_inlined() {
        let c = db();
        begin_run(&c, "run-3", None, None, None, None, None).unwrap();
        let h = Hop::new("c4:1", "llama.cpp")
            .with_scrape("nb-server-1-2", "/mnt/fast/scrape/x.json", 7826, "deadbeef")
            .scored(6.5, "pamp-v3");
        append_hop(&c, "run-3", &h).unwrap();
        let (rows, sha): (i64, String) = c
            .query_row(
                "SELECT scrape_rows, scrape_sha256 FROM run_hop WHERE run_id='run-3'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(rows, 7826);
        assert_eq!(sha, "deadbeef");
    }

    #[test]
    fn stale_open_runs_are_reported_not_closed() {
        let c = db();
        begin_run(&c, "run-4", None, None, None, None, None).unwrap();
        c.execute(
            "UPDATE run_envelope SET created_at = datetime('now','-2 hours') WHERE run_id='run-4'",
            [],
        )
        .unwrap();
        let stale = stale_runs(&c, 3600).unwrap();
        assert_eq!(stale, vec!["run-4".to_string()]);
        // still open: auto-closing would hide the lost card
        let st: String = c
            .query_row("SELECT status FROM run_envelope WHERE run_id='run-4'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(st, "open");
    }

    #[test]
    fn retention_roundtrip() {
        for r in [Retention::Divergence, Retention::FullLowScore, Retention::FullControl] {
            assert_eq!(Retention::parse(r.as_str()), r);
        }
    }
}
