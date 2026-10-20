//! SQLite histogram store for NautiBurn.
//!
//! The database is **the store**. The decision is the histogram. This crate
//! holds activation scrapes — ours and published — so belief can update when a
//! real gate disagrees. It does not schedule work, does not move weights, and
//! does not contain a model kernel.
//!
//! Two invariants shape the API:
//!
//! - **Histograms never mix models.** A `model_id` is weights + arch, never a
//!   hostname, and every count is keyed by it. Mixtral and Qwen cannot share a
//!   histogram even when they run on the same host.
//! - **Base and Instruct never share either.** Public traces are published for
//!   one weight set; [`HistDb::insert_scrape`] refuses a source that
//!   contradicts the `model_id` it was filed under.
//!
//! Hot / warm / cold is a *derived* query over
//! [`HistDb::histogram_for`], not a stored flag. Thresholds stay unlocked
//! until scrapes say otherwise; nothing in this crate hardcodes a count.

mod import;
mod schema;

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

pub use crate::import::{
    allenai_jsonl_paths, parse_continuum_counts, scrape_from_allenai_line, ContinuumCell,
    ContinuumCounts, ALLENAI_MIXTRAL_BASE_MODEL_ID, ALLENAI_OLMOE_MODEL_ID,
    CONTINUUM_MIXTRAL_INSTRUCT_MODEL_ID,
};
pub use crate::schema::{
    ActivationRow, BeliefRecord, HistogramCell, HistDbError, ModelRecord, ScrapeRecord,
    DB_SCHEMA_VERSION, MIGRATION_1, MIGRATION_2, MIGRATION_3, SCHEMA_VERSION,
};

/// Class used when a scrape arrives with no usable label.
///
/// The handoff is explicit: a missing class must not poison a real class, so
/// unknown rows land here rather than being merged into `code` or `prose`.
pub const UNKNOWN_CLASS: &str = "unknown";

/// One SQLite file holding activation scrapes, histograms, and beliefs.
pub struct HistDb {
    conn: Connection,
}

impl HistDb {
    /// Open (or create) a histogram store at `path`.
    ///
    /// Enables WAL so a reader (an agent, `sqlite3` on the shell) never blocks the
    /// scraper writing. Foreign keys are on so a row cannot reference a model that
    /// was never declared.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HistDbError> {
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// In-memory store, for tests and for a scratch histogram.
    pub fn open_in_memory() -> Result<Self, HistDbError> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self, HistDbError> {
        // busy_timeout absorbs a concurrent reader rather than failing the scrape.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        apply_migrations(&conn)?;

        let db = Self { conn };
        db.ensure_class(UNKNOWN_CLASS, "unknown")?;
        Ok(db)
    }

    /// The schema version recorded in `meta`.
    pub fn schema_version(&self) -> Result<i64, HistDbError> {
        let v: String = self.conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )?;
        v.parse().map_err(|_| HistDbError::SchemaVersion {
            got: -1,
            expected: SCHEMA_VERSION,
        })
    }

    fn ensure_class(&self, class_id: &str, label: &str) -> Result<(), HistDbError> {
        self.conn.execute(
            "INSERT INTO request_classes (class_id, label) VALUES (?1, ?2)
             ON CONFLICT(class_id) DO NOTHING",
            params![class_id, label],
        )?;
        Ok(())
    }

    /// Register (or update) a logical model. Safe to call repeatedly.
    pub fn upsert_model(&self, m: &ModelRecord) -> Result<(), HistDbError> {
        self.conn.execute(
            "INSERT INTO models (model_id, arch, n_layers, n_experts, top_k, hidden_size, notes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(model_id) DO UPDATE SET
               arch = excluded.arch,
               n_layers = excluded.n_layers,
               n_experts = excluded.n_experts,
               top_k = excluded.top_k,
               hidden_size = excluded.hidden_size,
               notes = excluded.notes",
            params![m.model_id, m.arch, m.n_layers, m.n_experts, m.top_k, m.hidden_size, m.notes],
        )?;
        Ok(())
    }

    /// Look up a declared model.
    pub fn model(&self, model_id: &str) -> Result<Option<ModelRecord>, HistDbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT model_id, arch, n_layers, n_experts, top_k, hidden_size, notes
                 FROM models WHERE model_id = ?1",
                params![model_id],
                |r| {
                    Ok(ModelRecord {
                        model_id: r.get(0)?,
                        arch: r.get(1)?,
                        n_layers: r.get(2)?,
                        n_experts: r.get(3)?,
                        top_k: r.get(4)?,
                        hidden_size: r.get(5)?,
                        notes: r.get(6)?,
                    })
                },
            )
            .optional()?)
    }

    /// Every registered model id, in insert order.
    pub fn model_ids(&self) -> Result<Vec<String>, HistDbError> {
        let mut stmt = self
            .conn
            .prepare("SELECT model_id FROM models ORDER BY model_id")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Request ids, oldest first. Used to seed a pre-prediction bar from
    /// prior scrapes without walking the request about to be named.
    pub fn request_ids(&self) -> Result<Vec<String>, HistDbError> {
        let mut stmt = self
            .conn
            .prepare("SELECT request_id FROM requests ORDER BY created_at, request_id")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Whether a request id is already stored.
    ///
    /// `request_id` is the `requests` primary key, so this is an index lookup
    /// rather than a scan — the duplicate check on `POST /scrape` uses this
    /// instead of reading every request id into memory.
    pub fn has_request(&self, request_id: &str) -> Result<bool, HistDbError> {
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM requests WHERE request_id = ?1 LIMIT 1")?;
        Ok(stmt
            .query_row(params![request_id], |_| Ok(()))
            .optional()
            .is_some())
    }

    /// Gate events for one `model_id`, optionally skipping one request.
    ///
    /// Pre-prediction seeds frequency / bigram from this so the named set
    /// does not leak the scrape it is about to be scored against.
    pub fn activations_for_model_except(
        &self,
        model_id: &str,
        except_request_id: Option<&str>,
    ) -> Result<Vec<ActivationEvent>, HistDbError> {
        let mut out = Vec::new();
        for id in self.request_ids()? {
            if except_request_id == Some(id.as_str()) {
                continue;
            }
            let req_model: Option<String> = self
                .conn
                .query_row(
                    "SELECT model_id FROM requests WHERE request_id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()?;
            if req_model.as_deref() != Some(model_id) {
                continue;
            }
            out.extend(self.activations_for(&id)?);
        }
        Ok(out)
    }

    /// Reject a `source` that cannot legitimately describe this `model_id`.
    ///
    /// This is the base/Instruct firewall from the handoff:
    ///
    /// - `allenai_mixtral` is **base** Mixtral-8x7B on C4, so it may not be
    ///   filed under any Instruct `model_id`.
    /// - `continuum_counts` is **Instruct** (`importance.activation_count.json`).
    ///
    /// Mixing them silently corrupts a histogram, and a histogram is exactly
    /// what the scheduler later trusts. Refuse loudly instead.
    fn check_source_model(&self, model_id: &str, source: &str) -> Result<(), HistDbError> {
        let m = self
            .model(model_id)?
            .ok_or_else(|| HistDbError::UnknownModel(model_id.to_string()))?;
        let is_instruct = m.model_id.to_ascii_lowercase().contains("instruct");

        if source == "allenai_mixtral" && is_instruct {
            return Err(HistDbError::SourceModelMismatch {
                model_id: model_id.to_string(),
                ingest_source: source.to_string(),
                reason: "allenai_mixtral publishes BASE Mixtral traces; use the base model_id",
            });
        }
        if source == "continuum_counts" && !is_instruct {
            return Err(HistDbError::SourceModelMismatch {
                model_id: model_id.to_string(),
                ingest_source: source.to_string(),
                reason: "continuum_counts is the INSTRUCT count histogram; use the Instruct model_id",
            });
        }
        Ok(())
    }

    /// Ingest one v1 scrape: request, tokens, activations, and the histogram
    /// bump. All in one transaction, so a rejected row leaves no partial
    /// request behind.
    pub fn insert_scrape(&mut self, rec: &ScrapeRecord) -> Result<(), HistDbError> {
        if rec.schema_version != SCHEMA_VERSION {
            return Err(HistDbError::SchemaVersion {
                got: rec.schema_version,
                expected: SCHEMA_VERSION,
            });
        }

        let model = self
            .model(&rec.model_id)?
            .ok_or_else(|| HistDbError::UnknownModel(rec.model_id.clone()))?;
        self.check_source_model(&rec.model_id, &rec.source)?;

        // Validate every row before opening the transaction.
        for row in &rec.activations {
            let got = row.expert_ids.len();
            if got as i64 != model.top_k {
                return Err(HistDbError::TopKMismatch {
                    model_id: rec.model_id.clone(),
                    got,
                    expected: model.top_k,
                });
            }
            for &e in &row.expert_ids {
                if e < 0 || e >= model.n_experts {
                    return Err(HistDbError::ExpertOutOfRange {
                        model_id: rec.model_id.clone(),
                        expert_id: e,
                        n_experts: model.n_experts,
                    });
                }
            }
        }

        // A missing class files under `unknown`; it must not overwrite a class.
        let class_id = rec.class_id.as_deref().unwrap_or(UNKNOWN_CLASS);
        self.ensure_class(class_id, class_id)?;

        let tx = self.conn.transaction()?;

        // Re-ingest of the same request_id must replace, not add. Subtract the
        // previous activation contribution before writing the new rows.
        {
            let mut stmt = tx.prepare(
                "SELECT r.model_id, r.class_id, a.phase, a.layer, a.expert_id
                 FROM activations a
                 JOIN requests r ON r.request_id = a.request_id
                 WHERE a.request_id = ?1",
            )?;
            let old: Vec<(String, String, String, i64, i64)> = stmt
                .query_map(params![rec.request_id], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })?
                .collect::<Result<_, _>>()?;
            drop(stmt);
            for (model_id, old_class, phase, layer, expert_id) in old {
                tx.execute(
                    "UPDATE histogram_counts
                     SET count = count - 1
                     WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3
                       AND layer = ?4 AND expert_id = ?5",
                    params![model_id, old_class, phase, layer, expert_id],
                )?;
            }
        }

        tx.execute(
            "INSERT INTO requests (request_id, model_id, class_id, source, phase_hint, n_tokens, dtype)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(request_id) DO UPDATE SET
               model_id = excluded.model_id,
               class_id = excluded.class_id,
               source = excluded.source,
               phase_hint = excluded.phase_hint,
               n_tokens = excluded.n_tokens,
               dtype = excluded.dtype",
            params![
                rec.request_id,
                rec.model_id,
                class_id,
                rec.source,
                rec.phase,
                rec.n_tokens,
                rec.dtype
            ],
        )?;

        if let Some(tokens) = &rec.token_ids {
            for (pos, &tok) in tokens.iter().enumerate() {
                tx.execute(
                    "INSERT INTO request_tokens (request_id, pos, token_id) VALUES (?1, ?2, ?3)
                     ON CONFLICT(request_id, pos) DO UPDATE SET token_id = excluded.token_id",
                    params![rec.request_id, pos as i64, tok],
                )?;
            }
        }

        for row in &rec.activations {
            for (rank, &expert) in row.expert_ids.iter().enumerate() {
                tx.execute(
                    "INSERT INTO activations
                       (request_id, phase, layer, token_pos, rank, expert_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                     ON CONFLICT(request_id, phase, layer, token_pos, rank)
                       DO UPDATE SET expert_id = excluded.expert_id",
                    params![
                        rec.request_id,
                        rec.phase,
                        row.layer,
                        row.token_pos,
                        rank as i64,
                        expert
                    ],
                )?;
            }
        }

        // Bump the histogram from the activations just written.
        for row in &rec.activations {
            for &expert in &row.expert_ids {
                tx.execute(
                    "INSERT INTO histogram_counts
                       (model_id, class_id, phase, layer, expert_id, count)
                     VALUES (?1, ?2, ?3, ?4, ?5, 1)
                     ON CONFLICT(model_id, class_id, phase, layer, expert_id)
                       DO UPDATE SET count = count + 1",
                    params![rec.model_id, class_id, rec.phase, row.layer, expert],
                )?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Read one histogram cell back. `None` when that expert never fired.
    pub fn histogram_count(
        &self,
        model_id: &str,
        class_id: &str,
        phase: &str,
        layer: i64,
        expert_id: i64,
    ) -> Result<Option<i64>, HistDbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT count FROM histogram_counts
                 WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3
                   AND layer = ?4 AND expert_id = ?5",
                params![model_id, class_id, phase, layer, expert_id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?)
    }

    /// Every non-zero cell for one `(model, class, phase)`, ordered by layer
    /// then expert.
    ///
    /// Hot / warm / cold is derived from this by the caller. Thresholds stay
    /// unlocked until scrapes say otherwise; nothing here hardcodes a count.
    pub fn histogram_for(
        &self,
        model_id: &str,
        class_id: &str,
        phase: &str,
    ) -> Result<Vec<HistogramCell>, HistDbError> {
        let mut stmt = self.conn.prepare(
            "SELECT layer, expert_id, count FROM histogram_counts
             WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3 AND count > 0
             ORDER BY layer, expert_id",
        )?;
        let rows = stmt.query_map(params![model_id, class_id, phase], |r| {
            Ok(HistogramCell {
                layer: r.get(0)?,
                expert_id: r.get(1)?,
                count: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Handoff query: cells for one `(model, class, phase, layer)`.
    pub fn query_histogram(
        &self,
        model_id: &str,
        class_id: &str,
        phase: &str,
        layer: i64,
    ) -> Result<Vec<HistogramCell>, HistDbError> {
        let mut stmt = self.conn.prepare(
            "SELECT layer, expert_id, count FROM histogram_counts
             WHERE model_id = ?1 AND class_id = ?2 AND phase = ?3 AND layer = ?4 AND count > 0
             ORDER BY expert_id",
        )?;
        let rows = stmt.query_map(params![model_id, class_id, phase, layer], |r| {
            Ok(HistogramCell {
                layer: r.get(0)?,
                expert_id: r.get(1)?,
                count: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Stored scrape dtype for a request (`f16`, …). `None` if missing or unset.
    pub fn request_dtype(&self, request_id: &str) -> Result<Option<String>, HistDbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT dtype FROM requests WHERE request_id = ?1",
                params![request_id],
                |r| r.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten())
    }

    /// How many rows are in `histogram_counts` (including zeros).
    pub fn histogram_row_count(&self) -> Result<i64, HistDbError> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM histogram_counts", [], |r| r.get(0))?)
    }

    /// One sample `histogram_counts` row, if any exist.
    pub fn histogram_sample(&self) -> Result<Option<HistogramCountRow>, HistDbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT model_id, class_id, phase, layer, expert_id, count
                 FROM histogram_counts
                 ORDER BY model_id, class_id, phase, layer, expert_id
                 LIMIT 1",
                [],
                |r| {
                    Ok(HistogramCountRow {
                        model_id: r.get(0)?,
                        class_id: r.get(1)?,
                        phase: r.get(2)?,
                        layer: r.get(3)?,
                        expert_id: r.get(4)?,
                        count: r.get(5)?,
                    })
                },
            )
            .optional()?)
    }

    /// Directly add to a histogram cell.
    ///
    /// This is the only path for aggregate-only imports (continuum
    /// `importance.activation_count.json`), which have counts but no per-token
    /// rows and therefore cannot validate a hit rate.
    pub fn bump_histogram(
        &self,
        model_id: &str,
        class_id: &str,
        phase: &str,
        layer: i64,
        expert_id: i64,
        delta: i64,
    ) -> Result<(), HistDbError> {
        if self.model(model_id)?.is_none() {
            return Err(HistDbError::UnknownModel(model_id.to_string()));
        }
        self.ensure_class(class_id, class_id)?;
        self.conn.execute(
            "INSERT INTO histogram_counts
               (model_id, class_id, phase, layer, expert_id, count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(model_id, class_id, phase, layer, expert_id)
               DO UPDATE SET count = count + excluded.count",
            params![model_id, class_id, phase, layer, expert_id, delta],
        )?;
        Ok(())
    }

    /// Record a predicted expert set before the gate.
    ///
    /// Returns the generated `belief_id`. Scoring the gate is a separate
    /// [`HistDb::record_disagreement`] call.
    pub fn record_belief(&mut self, rec: &BeliefRecord) -> Result<String, HistDbError> {
        let belief_id = uuid::Uuid::new_v4().to_string();
        let predicted_json = serde_json::to_string(&rec.expert_ids)?;
        self.conn.execute(
            "INSERT INTO beliefs (belief_id, request_id, layer, method, expert_ids, latency_us)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                belief_id,
                rec.request_id,
                rec.layer,
                rec.method,
                predicted_json,
                rec.latency_us
            ],
        )?;
        Ok(belief_id)
    }

    /// Score a belief against the experts the gate actually chose.
    ///
    /// Inserts `disagreements`. On a miss (predicted set ≠ actual set),
    /// increments `histogram_counts` for each actual expert id so one surprise
    /// updates belief. It does not move weight placement. Activation rows
    /// stay the scrape source of truth (no invented `token_pos`).
    ///
    /// Returns overlap `k` = |predicted ∩ actual|.
    pub fn record_disagreement(
        &mut self,
        belief_id: &str,
        actual_ids: &[i64],
    ) -> Result<usize, HistDbError> {
        self.record_disagreement_ex(belief_id, actual_ids, true)
    }

    /// Same as [`HistDb::record_disagreement`], but a shadow replay of an
    /// already-ingested scrape can set `bump_histogram = false` so four
    /// methods do not multiply the same actual ids into the bar.
    pub fn record_disagreement_ex(
        &mut self,
        belief_id: &str,
        actual_ids: &[i64],
        bump_histogram: bool,
    ) -> Result<usize, HistDbError> {
        let (request_id, layer, predicted_json): (String, i64, String) = self
            .conn
            .query_row(
                "SELECT request_id, layer, expert_ids FROM beliefs WHERE belief_id = ?1",
                params![belief_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?
            .ok_or_else(|| HistDbError::UnknownBelief(belief_id.to_string()))?;

        let predicted: Vec<i64> = serde_json::from_str(&predicted_json)?;
        let overlap = predicted.iter().filter(|e| actual_ids.contains(e)).count();
        let miss = {
            let mut a = predicted.clone();
            let mut b = actual_ids.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            a != b
        };
        let actual_json = serde_json::to_string(actual_ids)?;

        let req: Option<(String, String, Option<String>)> = self
            .conn
            .query_row(
                "SELECT model_id, class_id, phase_hint FROM requests WHERE request_id = ?1",
                params![request_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO disagreements (belief_id, layer, overlap_k, actual_ids)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(belief_id, layer) DO UPDATE SET
               overlap_k = excluded.overlap_k,
               actual_ids = excluded.actual_ids",
            params![belief_id, layer, overlap as i64, actual_json],
        )?;

        if miss && bump_histogram {
            if let Some((model_id, class_id, Some(phase))) = req {
                for &expert_id in actual_ids {
                    tx.execute(
                        "INSERT INTO histogram_counts
                           (model_id, class_id, phase, layer, expert_id, count)
                         VALUES (?1, ?2, ?3, ?4, ?5, 1)
                         ON CONFLICT(model_id, class_id, phase, layer, expert_id)
                           DO UPDATE SET count = count + 1",
                        params![model_id, class_id, phase, layer, expert_id],
                    )?;
                }
            }
        }

        tx.commit()?;
        Ok(overlap)
    }

    /// Overlap `k` recorded for one belief, if it was scored.
    pub fn belief_overlap(&self, belief_id: &str) -> Result<Option<usize>, HistDbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT overlap_k FROM disagreements WHERE belief_id = ?1",
                params![belief_id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
            .map(|v| v as usize))
    }

    /// Predict latency recorded on a belief, if present.
    pub fn belief_latency_us(&self, belief_id: &str) -> Result<Option<i64>, HistDbError> {
        Ok(self
            .conn
            .query_row(
                "SELECT latency_us FROM beliefs WHERE belief_id = ?1",
                params![belief_id],
                |r| r.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten())
    }

    /// How many belief rows are stored.
    pub fn belief_count(&self) -> Result<i64, HistDbError> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM beliefs", [], |r| r.get(0))?)
    }

    /// How many disagreement rows are stored.
    pub fn disagreement_count(&self) -> Result<i64, HistDbError> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM disagreements", [], |r| r.get(0))?)
    }

    /// Insert or replace a request row so beliefs can reference it.
    ///
    /// Does not write activations or bump the histogram. The shadow box uses
    /// this for a TSV run that is not a v1 scrape document.
    pub fn upsert_request(
        &self,
        request_id: &str,
        model_id: &str,
        class_id: &str,
        source: &str,
        phase_hint: Option<&str>,
        n_tokens: Option<i64>,
        dtype: Option<&str>,
    ) -> Result<(), HistDbError> {
        if self.model(model_id)?.is_none() {
            return Err(HistDbError::UnknownModel(model_id.to_string()));
        }
        self.ensure_class(class_id, class_id)?;
        self.conn.execute(
            "INSERT INTO requests (request_id, model_id, class_id, source, phase_hint, n_tokens, dtype)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(request_id) DO UPDATE SET
               model_id = excluded.model_id,
               class_id = excluded.class_id,
               source = excluded.source,
               phase_hint = excluded.phase_hint,
               n_tokens = excluded.n_tokens,
               dtype = excluded.dtype",
            params![
                request_id,
                model_id,
                class_id,
                source,
                phase_hint,
                n_tokens,
                dtype
            ],
        )?;
        Ok(())
    }

    /// Gate events for one request, ranks collapsed into expert-id lists.
    ///
    /// Order is `token_pos`, then `layer` — the causal walk the shadow box
    /// uses. Does not tokenize anything.
    pub fn activations_for(&self, request_id: &str) -> Result<Vec<ActivationEvent>, HistDbError> {
        let mut stmt = self.conn.prepare(
            "SELECT phase, layer, token_pos, rank, expert_id
             FROM activations
             WHERE request_id = ?1
             ORDER BY token_pos, layer, rank",
        )?;
        let rows = stmt.query_map(params![request_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;

        let mut events: Vec<ActivationEvent> = Vec::new();
        for row in rows {
            let (phase, layer, token_pos, _rank, expert_id) = row?;
            match events.last_mut() {
                Some(prev)
                    if prev.phase == phase
                        && prev.layer == layer
                        && prev.token_pos == token_pos =>
                {
                    prev.expert_ids.push(expert_id);
                }
                _ => events.push(ActivationEvent {
                    phase,
                    layer,
                    token_pos,
                    expert_ids: vec![expert_id],
                }),
            }
        }
        Ok(events)
    }

    /// Parse one v1 scrape JSON document and ingest it.
    pub fn insert_scrape_json(&mut self, json: &str) -> Result<(), HistDbError> {
        let rec: ScrapeRecord = serde_json::from_str(json)?;
        self.insert_scrape(&rec)
    }

    /// Parse many v1 scrape documents (one per JSON object, as a JSON array).
    ///
    /// Stops at the first bad document; the ones already committed stay, and
    /// the error names the index.
    pub fn insert_scrape_json_array(&mut self, json: &str) -> Result<usize, HistDbError> {
        let recs: Vec<ScrapeRecord> = serde_json::from_str(json)?;
        let n = recs.len();
        for (i, rec) in recs.iter().enumerate() {
            self.insert_scrape(rec).map_err(|e| {
                HistDbError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("[{i}]: {e}"),
                ))
            })?;
        }
        Ok(n)
    }

    /// Load a public trace directory that ships one JSON per row.
    ///
    /// Used for the AllenAI OLMoE / Mixtral dumps. Each file must be a v1
    /// scrape document; a file that fails stops the import and names the path.
    pub fn import_scrape_dir(&mut self, dir: impl AsRef<Path>) -> Result<usize, HistDbError> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        // Deterministic order so a failed import is reproducible.
        files.sort();

        let mut n = 0usize;
        for path in files {
            let text = std::fs::read_to_string(&path)?;
            self.insert_scrape_json(&text).map_err(|e| {
                HistDbError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{}: {e}", path.display()),
                ))
            })?;
            n += 1;
        }
        Ok(n)
    }

    /// AllenAI OLMoE C4 JSONL (`exp_ids` `[n_tokens][top_k][n_layers]`).
    ///
    /// `path` is a `.jsonl` file or a directory of them. Files that are already
    /// v1 scrape JSON are accepted. Uses base `olmoe-1b-7b-0924`.
    pub fn import_allenai_olmoe(&mut self, path: impl AsRef<Path>) -> Result<usize, HistDbError> {
        self.import_allenai_path(path, "allenai_olmoe", ALLENAI_OLMOE_MODEL_ID)
    }

    /// AllenAI Mixtral-base C4 JSONL. Never filed under an Instruct `model_id`.
    pub fn import_allenai_mixtral(&mut self, path: impl AsRef<Path>) -> Result<usize, HistDbError> {
        self.import_allenai_path(path, "allenai_mixtral", ALLENAI_MIXTRAL_BASE_MODEL_ID)
    }

    fn import_allenai_path(
        &mut self,
        path: impl AsRef<Path>,
        source: &str,
        default_model_id: &str,
    ) -> Result<usize, HistDbError> {
        let path = path.as_ref();
        let jsonl = allenai_jsonl_paths(path)?;
        if jsonl.is_empty() {
            return self.import_scrape_dir(path);
        }
        if self.model(default_model_id)?.is_none() {
            if let Some(m) = infer_model(default_model_id) {
                self.upsert_model(&m)?;
            } else {
                return Err(HistDbError::UnknownModel(default_model_id.to_string()));
            }
        }
        let mut n = 0usize;
        for file in jsonl {
            n += self.import_allenai_jsonl_file(&file, source, default_model_id)?;
        }
        Ok(n)
    }

    fn import_allenai_jsonl_file(
        &mut self,
        path: &Path,
        source: &str,
        default_model_id: &str,
    ) -> Result<usize, HistDbError> {
        let text = std::fs::read_to_string(path)?;
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("row");
        let mut n = 0usize;
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let request_id = format!("{source}:{stem}:{}", i + 1);
            let rec = scrape_from_allenai_line(
                line,
                request_id,
                default_model_id.to_string(),
                source.to_string(),
            )
            .map_err(|e| {
                HistDbError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{}:{}: {e}", path.display(), i + 1),
                ))
            })?;
            if self.model(&rec.model_id)?.is_none() {
                if let Some(m) = infer_model(&rec.model_id) {
                    self.upsert_model(&m)?;
                } else {
                    return Err(HistDbError::UnknownModel(rec.model_id));
                }
            }
            self.insert_scrape(&rec)?;
            n += 1;
        }
        Ok(n)
    }

    /// Continuum Instruct `importance.activation_count.json` → `histogram_counts`
    /// only (`source=continuum_counts`). No per-token rows.
    pub fn import_continuum_counts(&mut self, path: impl AsRef<Path>) -> Result<usize, HistDbError> {
        let parsed = parse_continuum_counts(&std::fs::read_to_string(path.as_ref())?)?;
        if self.model(&parsed.model_id)?.is_none() {
            if let Some(m) = infer_model(&parsed.model_id) {
                self.upsert_model(&m)?;
            } else {
                return Err(HistDbError::UnknownModel(parsed.model_id.clone()));
            }
        }
        self.check_source_model(&parsed.model_id, "continuum_counts")?;
        self.ensure_class(UNKNOWN_CLASS, "unknown")?;
        let mut n = 0usize;
        for cell in &parsed.cells {
            self.bump_histogram(
                &parsed.model_id,
                UNKNOWN_CLASS,
                "mixed",
                cell.layer,
                cell.expert_id,
                cell.count,
            )?;
            n += 1;
        }
        Ok(n)
    }

    /// Ingest one scrape file after upserting `model` if the id is unknown.
    pub fn ingest_scrape_file(
        &mut self,
        path: impl AsRef<Path>,
        model: Option<&ModelRecord>,
    ) -> Result<ScrapeRecord, HistDbError> {
        let rec: ScrapeRecord = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if self.model(&rec.model_id)?.is_none() {
            if let Some(m) = model {
                if m.model_id != rec.model_id {
                    return Err(HistDbError::UnknownModel(rec.model_id));
                }
                self.upsert_model(m)?;
            } else if let Some(inferred) = infer_model(&rec.model_id) {
                self.upsert_model(&inferred)?;
            } else {
                return Err(HistDbError::UnknownModel(rec.model_id));
            }
        }
        self.insert_scrape(&rec)?;
        Ok(rec)
    }
}

/// One gate event: experts the router chose at `(phase, layer, token_pos)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationEvent {
    pub phase: String,
    pub layer: i64,
    pub token_pos: i64,
    pub expert_ids: Vec<i64>,
}

/// One row of `histogram_counts`, for CLI / agent sample dumps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistogramCountRow {
    pub model_id: String,
    pub class_id: String,
    pub phase: String,
    pub layer: i64,
    pub expert_id: i64,
    pub count: i64,
}

/// Infer a `ModelRecord` for a well-known `model_id`.
///
/// OLMoE dims come from `nautiburn-engine::ModelProfile::OLMOE` (16 / 64 / 8 / 2048).
/// Mixtral 8x7B is 32 layers, 8 experts, top-2, hidden 4096.
pub fn infer_model(model_id: &str) -> Option<ModelRecord> {
    let id = model_id.to_ascii_lowercase();
    if id.contains("olmoe") {
        return Some(ModelRecord {
            model_id: model_id.to_string(),
            arch: "olmoe".to_string(),
            n_layers: 16,
            n_experts: 64,
            top_k: 8,
            hidden_size: 2048,
            notes: Some("inferred from olmoe model_id; dims match ModelProfile::OLMOE".into()),
        });
    }
    if id.contains("mixtral") {
        return Some(ModelRecord {
            model_id: model_id.to_string(),
            arch: "mixtral".to_string(),
            n_layers: 32,
            n_experts: 8,
            top_k: 2,
            hidden_size: 4096,
            notes: Some("inferred from mixtral model_id".into()),
        });
    }
    // Qwen3.6-35B-A3B / Tiel-Coder live GGUF (qwen3moe family).
    if id.contains("qwen3.6") || id.contains("qwen36") || id.contains("tiel") {
        return Some(ModelRecord {
            model_id: model_id.to_string(),
            arch: "qwen3moe".to_string(),
            n_layers: 40,
            n_experts: 256,
            top_k: 8,
            hidden_size: 2048,
            notes: Some("inferred from qwen3.6 / tiel model_id".into()),
        });
    }
    // KAT-Coder qwen35moe GGUF (256 experts / top-8 / 40 layers).
    if id.contains("kat") || id.contains("qwen35moe") || id.contains("qwen3.5-moe") {
        return Some(ModelRecord {
            model_id: model_id.to_string(),
            arch: "qwen35moe".to_string(),
            n_layers: 40,
            n_experts: 256,
            top_k: 8,
            hidden_size: 2048,
            notes: Some("inferred from kat / qwen35moe model_id".into()),
        });
    }
    // Gemma-4 MoE GGUFs (gemma-4-26b-moe / gemma-4-26b-a4b sidechannel scrapes):
    // 30 MoE layers, 128 experts, top-8, hidden 2816 (gemma4.* keys in the GGUF header).
    if id.contains("gemma") && (id.contains("moe") || id.contains("a4b")) {
        return Some(ModelRecord {
            model_id: model_id.to_string(),
            arch: "gemma4moe".to_string(),
            n_layers: 30,
            n_experts: 128,
            top_k: 8,
            hidden_size: 2816,
            notes: Some("inferred from gemma-4 moe/a4b model_id; dims from the GGUF header".into()),
        });
    }
    // Qwen1.5-MoE-A2.7B chat GGUF: 24 routed layers, 60 experts, top-4, hidden 2048 (qwen2moe.* keys).
    if id.contains("qwen1.5-moe") || id.contains("qwen15-moe") {
        return Some(ModelRecord {
            model_id: model_id.to_string(),
            arch: "qwen15moe".to_string(),
            n_layers: 24,
            n_experts: 60,
            top_k: 4,
            hidden_size: 2048,
            notes: Some("inferred from qwen1.5-moe model_id; dims from the GGUF header".into()),
        });
    }
    None
}

fn apply_migrations(conn: &Connection) -> Result<(), HistDbError> {
    let existing = stored_schema_version(conn)?;
    let mut got = match existing.as_deref() {
        None => {
            conn.execute_batch(MIGRATION_1)?;
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', '1')",
                [],
            )?;
            1_i64
        }
        Some(v) => v.parse().map_err(|_| HistDbError::SchemaVersion {
            got: -1,
            expected: DB_SCHEMA_VERSION,
        })?,
    };
    if got > DB_SCHEMA_VERSION {
        return Err(HistDbError::SchemaVersion {
            got,
            expected: DB_SCHEMA_VERSION,
        });
    }
    if got < 2 {
        conn.execute_batch(MIGRATION_2)?;
        conn.execute(
            "UPDATE meta SET value = '2' WHERE key = 'schema_version'",
            [],
        )?;
        got = 2;
    }
    if got < 3 {
        conn.execute_batch(MIGRATION_3)?;
        conn.execute(
            "UPDATE meta SET value = '3' WHERE key = 'schema_version'",
            [],
        )?;
        got = 3;
    }
    let _ = got;
    Ok(())
}

fn stored_schema_version(conn: &Connection) -> Result<Option<String>, HistDbError> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'meta'",
        [],
        |r| r.get(0),
    )?;
    if exists == 0 {
        return Ok(None);
    }
    Ok(conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .optional()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_model(model_id: &str) -> ModelRecord {
        ModelRecord {
            model_id: model_id.to_string(),
            arch: "toy".to_string(),
            n_layers: 2,
            n_experts: 4,
            top_k: 2,
            hidden_size: 8,
            notes: None,
        }
    }

    fn scrape(model_id: &str, request_id: &str, source: &str) -> ScrapeRecord {
        ScrapeRecord {
            schema_version: 1,
            model_id: model_id.to_string(),
            request_id: request_id.to_string(),
            class_id: Some("code".to_string()),
            source: source.to_string(),
            phase: "prefill".to_string(),
            n_tokens: Some(2),
            dtype: Some("f16".to_string()),
            activations: vec![
                ActivationRow {
                    layer: 0,
                    token_pos: 0,
                    expert_ids: vec![1, 0],
                },
                ActivationRow {
                    layer: 0,
                    token_pos: 1,
                    expert_ids: vec![1, 2],
                },
            ],
            token_ids: Some(vec![11, 12]),
        }
    }

    #[test]
    fn has_request_detects_ingested_request() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();
        assert!(db.has_request("req-1").unwrap());
        assert!(!db.has_request("req-2").unwrap());
    }

    #[test]
    fn empty_db_ingest_updates_histogram_counts() {
        let mut db = HistDb::open_in_memory().unwrap();
        assert_eq!(db.schema_version().unwrap(), DB_SCHEMA_VERSION);
        assert_eq!(db.histogram_row_count().unwrap(), 0);

        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();
        assert_eq!(db.request_dtype("req-1").unwrap().as_deref(), Some("f16"));

        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap(), Some(2));
        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 0).unwrap(), Some(1));
        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 2).unwrap(), Some(1));
        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 3).unwrap(), None);

        let cells = db.query_histogram("toy-moe", "code", "prefill", 0).unwrap();
        assert_eq!(cells.len(), 3);
        assert_eq!(db.histogram_row_count().unwrap(), 3);
    }

    #[test]
    fn refuse_mixtral_base_rows_under_instruct_model_id() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&ModelRecord {
            model_id: "mixtral-8x7b-instruct-q4_k_m".into(),
            arch: "mixtral".into(),
            n_layers: 32,
            n_experts: 8,
            top_k: 2,
            hidden_size: 4096,
            notes: None,
        })
        .unwrap();

        let err = db
            .insert_scrape(&scrape(
                "mixtral-8x7b-instruct-q4_k_m",
                "req-base",
                "allenai_mixtral",
            ))
            .unwrap_err();
        match err {
            HistDbError::SourceModelMismatch { model_id, ingest_source, .. } => {
                assert!(model_id.contains("instruct"));
                assert_eq!(ingest_source, "allenai_mixtral");
            }
            other => panic!("expected SourceModelMismatch, got {other}"),
        }
        assert_eq!(db.histogram_row_count().unwrap(), 0);
    }

    #[test]
    fn fp16_scrape_json_v1_deserializes() {
        let json = r#"{
            "schema_version": 1,
            "model_id": "olmoe-1b-7b-0924-f16",
            "request_id": "nb-fp16-test",
            "class_id": "olmoe-layer0-forced-tile",
            "source": "scrape",
            "phase": "prefill",
            "n_tokens": 1,
            "dtype": "f16",
            "activations": [{"layer": 0, "token_pos": 0, "expert_ids": [1,2,3,4,5,6,7,8]}],
            "token_ids": [1]
        }"#;
        let rec: ScrapeRecord = serde_json::from_str(json).unwrap();
        assert_eq!(rec.dtype.as_deref(), Some("f16"));
        assert_eq!(rec.activations[0].expert_ids.len(), 8);
        let inferred = infer_model(&rec.model_id).unwrap();
        assert_eq!(inferred.n_experts, 64);
        assert_eq!(inferred.top_k, 8);
        assert_eq!(inferred.hidden_size, 2048);
        assert_eq!(inferred.n_layers, 16);
    }

    #[test]
    fn activations_for_model_except_skips_named_request() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-2", "scrape")).unwrap();
        let ids = db.request_ids().unwrap();
        assert_eq!(ids, vec!["req-1".to_string(), "req-2".to_string()]);
        let prior = db
            .activations_for_model_except("toy-moe", Some("req-2"))
            .unwrap();
        assert_eq!(prior.len(), 2);
        assert!(prior.iter().all(|e| e.expert_ids == vec![1, 0] || e.expert_ids == vec![1, 2]));
        let none = db
            .activations_for_model_except("other", None)
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn scrape_json_defaults_source_to_scrape() {
        let json = r#"{
            "schema_version": 1,
            "model_id": "toy-moe",
            "request_id": "req-min",
            "phase": "prefill",
            "dtype": "f16",
            "activations": [{"layer": 0, "token_pos": 0, "expert_ids": [1, 0]}]
        }"#;
        let rec: ScrapeRecord = serde_json::from_str(json).unwrap();
        assert_eq!(rec.source, "scrape");
        assert_eq!(rec.dtype.as_deref(), Some("f16"));
        assert!(rec.class_id.is_none());
    }

    #[test]
    fn reingest_same_request_does_not_double_count() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&toy_model("toy-moe")).unwrap();
        let rec = scrape("toy-moe", "req-1", "scrape");
        db.insert_scrape(&rec).unwrap();
        db.insert_scrape(&rec).unwrap();
        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap(), Some(2));
        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 0).unwrap(), Some(1));
        assert_eq!(db.histogram_count("toy-moe", "code", "prefill", 0, 2).unwrap(), Some(1));
    }

    #[test]
    fn disagreement_miss_increments_histogram_counts() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();

        let before_1 = db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap();
        let before_0 = db.histogram_count("toy-moe", "code", "prefill", 0, 0).unwrap();

        let belief_id = db
            .record_belief(&BeliefRecord {
                request_id: "req-1".into(),
                layer: 0,
                method: "frequency".into(),
                expert_ids: vec![3, 2],
                latency_us: Some(12),
            })
            .unwrap();
        let overlap = db.record_disagreement(&belief_id, &[1, 0]).unwrap();
        assert_eq!(overlap, 0);
        assert_eq!(db.belief_overlap(&belief_id).unwrap(), Some(0));

        assert_eq!(
            db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap(),
            before_1.map(|c| c + 1)
        );
        assert_eq!(
            db.histogram_count("toy-moe", "code", "prefill", 0, 0).unwrap(),
            before_0.map(|c| c + 1)
        );
    }

    #[test]
    fn shadow_disagreement_can_skip_histogram_bump() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();
        let before = db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap();

        let belief_id = db
            .record_belief(&BeliefRecord {
                request_id: "req-1".into(),
                layer: 0,
                method: "frequency".into(),
                expert_ids: vec![3, 2],
                latency_us: Some(4),
            })
            .unwrap();
        let overlap = db
            .record_disagreement_ex(&belief_id, &[1, 0], false)
            .unwrap();
        assert_eq!(overlap, 0);
        assert_eq!(db.belief_latency_us(&belief_id).unwrap(), Some(4));
        assert_eq!(
            db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap(),
            before
        );
        let events = db.activations_for("req-1").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].expert_ids, vec![1, 0]);
    }

    #[test]
    fn full_overlap_is_not_a_miss() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();
        let before = db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap();

        let belief_id = db
            .record_belief(&BeliefRecord {
                request_id: "req-1".into(),
                layer: 0,
                method: "frequency".into(),
                expert_ids: vec![1, 0],
                latency_us: None,
            })
            .unwrap();
        let overlap = db.record_disagreement(&belief_id, &[1, 0]).unwrap();
        assert_eq!(overlap, 2);
        assert_eq!(
            db.histogram_count("toy-moe", "code", "prefill", 0, 1).unwrap(),
            before
        );
    }

    #[test]
    fn allenai_olmoe_jsonl_exp_ids_updates_histogram() {
        let dir = std::env::temp_dir().join(format!("histodb-olmoe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // 2 tokens × top-8 × 2 layers. Shape from the handoff / AllenAI script.
        let line = r#"{"input_ids":[1,2],"exp_ids":[[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,0]],[[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,8]]]}"#;
        std::fs::write(dir.join("c4_results.jsonl"), format!("{line}\n")).unwrap();

        let mut db = HistDb::open_in_memory().unwrap();
        let n = db.import_allenai_olmoe(&dir).unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            db.histogram_count(ALLENAI_OLMOE_MODEL_ID, "c4", "prefill", 0, 0)
                .unwrap(),
            Some(1)
        );
        assert_eq!(
            db.histogram_count(ALLENAI_OLMOE_MODEL_ID, "c4", "prefill", 1, 1)
                .unwrap(),
            Some(1)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn allenai_mixtral_jsonl_uses_base_model_id() {
        let dir = std::env::temp_dir().join(format!("histodb-mixtral-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // 2 tokens × top-2 × 2 layers.
        let line = r#"{"input_ids":[1,2],"exp_ids":[[[0,1],[1,0]],[[2,3],[3,2]]]}"#;
        std::fs::write(dir.join("c4_results.jsonl"), format!("{line}\n")).unwrap();

        let mut db = HistDb::open_in_memory().unwrap();
        db.import_allenai_mixtral(&dir).unwrap();
        let ids = db.model_ids().unwrap();
        assert!(ids.iter().any(|id| id == ALLENAI_MIXTRAL_BASE_MODEL_ID));
        assert!(!ids.iter().any(|id| id.to_ascii_lowercase().contains("instruct")));
        assert_eq!(
            db.histogram_count(ALLENAI_MIXTRAL_BASE_MODEL_ID, "c4", "prefill", 0, 0)
                .unwrap(),
            Some(1)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn continuum_counts_seed_histogram_only() {
        let path = std::env::temp_dir().join(format!(
            "histodb-continuum-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","activation_counts":{"0":[10,0,0,0,0,0,0,1]},"num_experts":8,"num_experts_per_tok":2}"#,
        )
        .unwrap();

        let mut db = HistDb::open_in_memory().unwrap();
        let n = db.import_continuum_counts(&path).unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            db.histogram_count(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                UNKNOWN_CLASS,
                "mixed",
                0,
                0
            )
            .unwrap(),
            Some(10)
        );
        assert_eq!(
            db.histogram_count(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                UNKNOWN_CLASS,
                "mixed",
                0,
                7
            )
            .unwrap(),
            Some(1)
        );
        let act: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM activations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(act, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn continuum_refuses_base_mixtral_model_id() {
        let mut db = HistDb::open_in_memory().unwrap();
        db.upsert_model(&infer_model("mixtral-8x7b-v0.1").unwrap()).unwrap();
        let err = db
            .import_continuum_counts_for_test("mixtral-8x7b-v0.1", "{\"activation_counts\":{\"0\":[1,0,0,0,0,0,0,0]}}")
            .unwrap_err();
        match err {
            HistDbError::SourceModelMismatch { ingest_source, .. } => {
                assert_eq!(ingest_source, "continuum_counts");
            }
            other => panic!("expected SourceModelMismatch, got {other}"),
        }
    }
}

impl HistDb {
    #[cfg(test)]
    fn import_continuum_counts_for_test(
        &mut self,
        model_id: &str,
        json: &str,
    ) -> Result<usize, HistDbError> {
        let mut parsed = parse_continuum_counts(json)?;
        parsed.model_id = model_id.to_string();
        self.check_source_model(&parsed.model_id, "continuum_counts")?;
        self.ensure_class(UNKNOWN_CLASS, "unknown")?;
        let mut n = 0usize;
        for cell in &parsed.cells {
            self.bump_histogram(
                &parsed.model_id,
                UNKNOWN_CLASS,
                "mixed",
                cell.layer,
                cell.expert_id,
                cell.count,
            )?;
            n += 1;
        }
        Ok(n)
    }
}
