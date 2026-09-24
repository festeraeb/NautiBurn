//! One schema file for the job record and the two log payloads.
//!
//! `ExpertId` is a weight-slice index inside one forward.
//! `SpecialistId` is a model id for a process. They are different types.
//! A job record has no expert-id field.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Index of one expert slab. Not a model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExpertId(pub u16);

/// Identity of a specialist process. Not an expert slab index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpecialistId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    Code,
    Json,
    Prose,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lease {
    pub id: String,
    pub expires_at_unix_s: i64,
}

/// v1 job record. No expert id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobRecord {
    pub mission_id: String,
    pub task_class: TaskClass,
    pub language: String,
    pub token_estimate: u32,
    pub tools: Vec<String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub lease: Lease,
    pub device_affinity: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertDispatch {
    GpuResident,
    CpuMiss,
    PrefetchHit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialistDispatch {
    Preloaded,
    PolisherMiss,
}

/// Expert log line. `expert_ids` are [`ExpertId`], never a [`SpecialistId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertLog {
    pub timestamp_unix_s: i64,
    pub machine: String,
    pub device_uuid: String,
    pub backend: String,
    pub dtype: String,
    pub phase: Phase,
    pub layer: u32,
    pub expert_ids: Vec<ExpertId>,
    pub token_id: u32,
    pub arch: String,
    pub quant: String,
    pub predicted_ids: Vec<ExpertId>,
    pub hit: bool,
    pub dispatch: ExpertDispatch,
}

/// Specialist log line. `model_id` is a [`SpecialistId`], never an [`ExpertId`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialistLog {
    pub timestamp_unix_s: i64,
    pub machine: String,
    pub device_uuid: String,
    pub backend: String,
    pub dtype: String,
    pub mission_id: String,
    pub role: String,
    pub model_id: SpecialistId,
    pub quant: String,
    pub tools_required: Vec<String>,
    pub tools_ready_unix_s: i64,
    pub grade: String,
    pub stance: Option<String>,
    pub dispatch: SpecialistDispatch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogRecord {
    Expert(ExpertLog),
    Specialist(SpecialistLog),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    MissingField(String),
    UnknownTaskClass(String),
    ExpiredLease { expires_at_unix_s: i64, now_unix_s: i64 },
    SmuggledField(String),
    Json(String),
}

pub fn parse_job(json: &str, now_unix_s: i64) -> Result<JobRecord, SchemaError> {
    let job: JobRecord = from_json(json)?;
    if job.lease.expires_at_unix_s <= now_unix_s {
        return Err(SchemaError::ExpiredLease {
            expires_at_unix_s: job.lease.expires_at_unix_s,
            now_unix_s,
        });
    }
    Ok(job)
}

pub fn parse_log(json: &str) -> Result<LogRecord, SchemaError> {
    from_json(json)
}

fn from_json<T: DeserializeOwned>(json: &str) -> Result<T, SchemaError> {
    serde_json::from_str(json).map_err(classify)
}

fn classify(err: serde_json::Error) -> SchemaError {
    let msg = err.to_string();
    if let Some(field) = quoted_after(&msg, "missing field") {
        return SchemaError::MissingField(field);
    }
    if let Some(field) = quoted_after(&msg, "unknown field") {
        return SchemaError::SmuggledField(field);
    }
    if msg.contains("unknown variant") {
        if let Some(variant) = quoted_after(&msg, "unknown variant") {
            if msg.contains("`code`") {
                return SchemaError::UnknownTaskClass(variant);
            }
            return SchemaError::Json(msg);
        }
    }
    if msg.contains("invalid type") {
        if let Some(field) = field_at(&msg) {
            return SchemaError::SmuggledField(field);
        }
    }
    SchemaError::Json(msg)
}

fn quoted_after(msg: &str, needle: &str) -> Option<String> {
    let rest = msg.split(needle).nth(1)?;
    let start = rest.find('`')? + 1;
    let end = rest[start..].find('`')? + start;
    Some(rest[start..end].to_string())
}

fn field_at(msg: &str) -> Option<String> {
    let rest = msg.split("at ").nth(1)?;
    let name = rest.trim().trim_end_matches('.');
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000;

    const GOLDEN_JOB: &str = include_str!("../fixtures/golden_job.json");
    const MISSING_FIELD: &str = include_str!("../fixtures/reject_missing_field.json");
    const UNKNOWN_CLASS: &str = include_str!("../fixtures/reject_unknown_task_class.json");
    const EXPIRED_LEASE: &str = include_str!("../fixtures/reject_expired_lease.json");
    const SMUGGLED_JOB: &str = include_str!("../fixtures/reject_expert_id_smuggled.json");
    const GOLDEN_EXPERT: &str = include_str!("../fixtures/golden_expert.json");
    const GOLDEN_SPECIALIST: &str = include_str!("../fixtures/golden_specialist.json");
    const EXPERT_SMUGGLES_MODEL: &str = include_str!("../fixtures/reject_expert_has_model_id.json");
    const SPECIALIST_SMUGGLES_EXPERT: &str =
        include_str!("../fixtures/reject_specialist_has_expert_id.json");

    #[test]
    fn golden_job_is_accepted() {
        let job = parse_job(GOLDEN_JOB, NOW).expect("golden job");
        assert_eq!(job.mission_id, "mission-b2-golden");
        assert_eq!(job.task_class, TaskClass::Code);
        assert_eq!(job.tools, vec!["cargo_check".to_string()]);
        assert!(job.lease.expires_at_unix_s > NOW);
    }

    #[test]
    fn rejects_missing_field() {
        let err = parse_job(MISSING_FIELD, NOW).expect_err("missing field");
        assert_eq!(err, SchemaError::MissingField("mission_id".into()));
    }

    #[test]
    fn rejects_unknown_task_class() {
        let err = parse_job(UNKNOWN_CLASS, NOW).expect_err("unknown class");
        assert_eq!(err, SchemaError::UnknownTaskClass("router".into()));
    }

    #[test]
    fn rejects_expired_lease() {
        let err = parse_job(EXPIRED_LEASE, NOW).expect_err("expired");
        assert_eq!(
            err,
            SchemaError::ExpiredLease {
                expires_at_unix_s: 1_600_000_000,
                now_unix_s: NOW,
            }
        );
    }

    #[test]
    fn rejects_expert_id_smuggled_into_job() {
        let err = parse_job(SMUGGLED_JOB, NOW).expect_err("smuggled");
        assert_eq!(err, SchemaError::SmuggledField("expert_id".into()));
    }

    #[test]
    fn golden_expert_and_specialist_logs_are_accepted() {
        let expert = parse_log(GOLDEN_EXPERT).expect("expert");
        let specialist = parse_log(GOLDEN_SPECIALIST).expect("specialist");
        match expert {
            LogRecord::Expert(row) => {
                assert_eq!(row.expert_ids, vec![ExpertId(0), ExpertId(7)]);
                assert_eq!(row.dispatch, ExpertDispatch::GpuResident);
                assert_eq!(row.phase, Phase::Prefill);
            }
            LogRecord::Specialist(_) => panic!("expert golden parsed as specialist"),
        }
        match specialist {
            LogRecord::Specialist(row) => {
                assert_eq!(row.model_id, SpecialistId("qwen2.5-coder-7b".into()));
                assert_eq!(row.dispatch, SpecialistDispatch::Preloaded);
                assert!(row.stance.is_none());
            }
            LogRecord::Expert(_) => panic!("specialist golden parsed as expert"),
        }
    }

    #[test]
    fn rejects_model_id_smuggled_into_expert_log() {
        let err = parse_log(EXPERT_SMUGGLES_MODEL).expect_err("model in expert");
        assert_eq!(err, SchemaError::SmuggledField("model_id".into()));
    }

    #[test]
    fn rejects_expert_id_smuggled_into_specialist_log() {
        let err = parse_log(SPECIALIST_SMUGGLES_EXPERT).expect_err("expert in specialist");
        assert_eq!(err, SchemaError::SmuggledField("expert_id".into()));
    }

    #[test]
    fn specialist_dispatch_does_not_fit_an_expert_log() {
        let mut value: serde_json::Value = serde_json::from_str(GOLDEN_EXPERT).unwrap();
        value["dispatch"] = serde_json::Value::String("preloaded".into());
        let err = parse_log(&value.to_string()).expect_err("wrong dispatch");
        assert!(matches!(err, SchemaError::Json(_)));
    }

    /// `SpecialistId` does not implement the expert-slot marker.
    #[test]
    fn id_types_do_not_unify() {
        fn accepts_expert<T: ExpertSlot>(id: T) -> u16 {
            id.as_expert()
        }
        assert_eq!(accepts_expert(ExpertId(3)), 3);
        let _specialist = SpecialistId("not-an-expert".into());
        // `accepts_expert(_specialist)` does not compile: SpecialistId is not ExpertSlot.
    }
}

/// Marker for the expert slot. [`SpecialistId`] does not implement this.
pub trait ExpertSlot {
    fn as_expert(self) -> u16;
}

impl ExpertSlot for ExpertId {
    fn as_expert(self) -> u16 {
        self.0
    }
}
