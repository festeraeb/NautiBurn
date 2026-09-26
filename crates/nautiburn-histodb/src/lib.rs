mod schema;

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

pub use crate::schema::{
    ActivationRow, BeliefRecord, HistogramCell, HistDbError, ModelRecord, ScrapeRecord,
    MIGRATION_1, SCHEMA_VERSION,
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

        let existing = stored_schema_version(&conn)?;
        match existing.as_deref() {
            None => {
                conn.execute_batch(MIGRATION_1)?;
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)",
                    params![SCHEMA_VERSION.to_string()],
                )?;
            }
            Some(v) => {
                let got: i64 = v.parse().map_err(|_| HistDbError::SchemaVersion {
                    got: -1,
                    expected: SCHEMA_VERSION,
                })?;
                if got != SCHEMA_VERSION {
                    return Err(HistDbError::SchemaVersion {
                        got,
                        expected: SCHEMA_VERSION,
                    });
                }
            }
        }

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
        let is_instruct = m.model_id.contains("instruct");

        if source == "allenai_mixtral" && is_instruct {
            return Err(HistDbError::SourceModelMismatch {
                model_id: model_id.to_string(),
                source: source.to_string(),
                reason: "allenai_mixtral publishes BASE Mixtral traces; use the base model_id",
            });
        }
        if source == "continuum_counts" && !is_instruct {
            return Err(HistDbError::SourceModelMismatch {
                model_id: model_id.to_string(),
                source: source.to_string(),
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

        tx.execute(
            "INSERT INTO requests (request_id, model_id, class_id, source, phase_hint, n_tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(request_id) DO UPDATE SET
               model_id = excluded.model_id,
               class_id = excluded.class_id,
               source = excluded.source,
               phase_hint = excluded.phase_hint,
               n_tokens = excluded.n_tokens",
            params![
                rec.request_id,
                rec.model_id,
                class_id,
                rec.source,
                rec.phase,
                rec.n_tokens
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

    /// Record a predicted expert set, and score it against what the gate
    /// actually did.
    ///
    /// Returns the generated `belief_id` and the overlap `k`. The actual ids
    /// are *kept* in `disagreements` and already in `activations`; a miss
    /// updates belief counts, it does not move weight placement.
    pub fn record_belief(
        &mut self,
        rec: &BeliefRecord,
        actual_ids: &[i64],
    ) -> Result<(String, usize), HistDbError> {
        let belief_id = uuid::Uuid::new_v4().to_string();
        let overlap = rec
            .expert_ids
            .iter()
            .filter(|e| actual_ids.contains(e))
            .count();
        let actual_json = serde_json::to_string(actual_ids)?;
        let predicted_json = serde_json::to_string(&rec.expert_ids)?;

        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO beliefs (belief_id, request_id, layer, method, expert_ids)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                belief_id,
                rec.request_id,
                rec.layer,
                rec.method,
                predicted_json
            ],
        )?;
        tx.execute(
            "INSERT INTO disagreements (belief_id, layer, overlap_k, actual_ids)
             VALUES (?1, ?2, ?3, ?4)",
            params![belief_id, rec.layer, overlap as i64, actual_json],
        )?;
        tx.commit()?;
        Ok((belief_id, overlap))
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

    /// AllenAI OLMoE directory import. Files must already be v1 scrape JSON
    /// with `source=allenai_olmoe`.
    pub fn import_allenai_olmoe(&mut self, dir: impl AsRef<Path>) -> Result<usize, HistDbError> {
        self.import_scrape_dir(dir)
    }

    /// AllenAI Mixtral-base directory import. Files must already be v1 scrape
    /// JSON with `source=allenai_mixtral` on a **base** `model_id`.
    pub fn import_allenai_mixtral(&mut self, dir: impl AsRef<Path>) -> Result<usize, HistDbError> {
        self.import_scrape_dir(dir)
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
    None
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
    fn empty_db_ingest_updates_histogram_counts() {
        let mut db = HistDb::open_in_memory().unwrap();
        assert_eq!(db.schema_version().unwrap(), 1);
        assert_eq!(db.histogram_row_count().unwrap(), 0);

        db.upsert_model(&toy_model("toy-moe")).unwrap();
        db.insert_scrape(&scrape("toy-moe", "req-1", "scrape")).unwrap();

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
            HistDbError::SourceModelMismatch { model_id, source, .. } => {
                assert!(model_id.contains("instruct"));
                assert_eq!(source, "allenai_mixtral");
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
}
