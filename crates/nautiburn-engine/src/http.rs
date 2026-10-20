//! Brief 3 — HTTP API for nautiburn-engine.
//!
//! Thin axum surface over the existing [`Engine`]. It owns no routing logic of
//! its own: every endpoint delegates to the same `Engine` methods the CLI uses,
//! so a request answered here and a request answered by `nautiburn-engine route`
//! cannot disagree.
//!
//! Endpoints (bound 127.0.0.1:9119):
//!   POST /route    ExpertPass JSON -> NextExperts JSON
//!   POST /scrape   ScrapeDocument JSON -> write_scrape() result
//!   GET  /health   "ok"
//!   GET  /stats    engine counters as JSON
//!
//! Deliberately NOT here (out of scope for Brief 3, tracked elsewhere):
//! speculative decode, the OpenVINO/CPU path, and the layer0 CubeCL bridge.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;

use nautiburn_histodb::{
    ActivationRow as HistActivationRow, HistDb, ScrapeRecord, SCHEMA_VERSION,
};

use crate::scrape::{write_scrape, ScrapeDocument};
use crate::{Engine, ExpertPass, ModelProfile};

/// Shared by every handler. The CLI builds its own `Engine` per invocation, so
/// there is exactly one instance here for the server's lifetime.
pub type Shared = Arc<Mutex<Engine>>;

pub fn shared(profile: ModelProfile) -> Shared {
    Arc::new(Mutex::new(Engine::new(profile)))
}

#[derive(Clone)]
struct AppState {
    engine: Shared,
    db: Option<Arc<StdMutex<HistDb>>>,
}

pub fn router(state: Shared) -> Router {
    router_with_histodb(state, None)
}

pub fn router_with_histodb(
    state: Shared,
    db: Option<Arc<StdMutex<HistDb>>>,
) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/route", post(route))
        .route("/scrape", post(scrape))
        .with_state(AppState { engine: state, db })
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let engine = state.engine.lock();
    // Read the engine's own public fields and the crate-level helpers, so these
    // are the same numbers the CLI prints rather than a second source of truth.
    let p = &engine.profile;
    let body = serde_json::json!({
        "profile": p.name,
        "layers": p.layers,
        "experts": p.experts,
        "k": p.k,
        "activation_bytes_1tok": p.activation_bytes(1),
        "slab_bytes_all_experts": p.slab_bytes(p.experts),
        "scheduled_compute_dtype": crate::scheduled_compute_dtype(),
        "observations": engine.observations.len(),
        "fp16_launch": match crate::admit_fp16_on_p100(4096) {
            crate::Fp16Launch::Admitted => "admitted".to_string(),
            crate::Fp16Launch::Refused { shared_bytes, cap_bytes } =>
                format!("refused shared={shared_bytes} cap={cap_bytes}"),
        },
    });
    (StatusCode::OK, Json(body))
}

async fn route(
    State(state): State<AppState>,
    body: Result<Json<ExpertPass>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let Json(pass) = match body {
        Ok(j) => j,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": rej.body_text() })),
            )
        }
    };
    let engine = state.engine.lock();
    let next = engine.route(&pass);
    // A serialization failure here would otherwise be swallowed by
    // unwrap_or_default() and answered with 200 "{}" — a client treating 2xx as
    // success would proceed with an empty route. Surface it as a 500 instead.
    match serde_json::to_value(next) {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("serialize NextExperts: {err}") })),
        ),
    }
}

/// Body for POST /scrape: the engine's own document plus where to put it.
///
/// `out` is required. Writing to an implicit directory would mean the server
/// picks a path the operator never chose, and the CLI already takes the path as
/// an argument — so the HTTP path takes it explicitly and stays comparable.
#[derive(Debug, serde::Deserialize)]
struct ScrapeRequest {
    out: String,
    #[serde(flatten)]
    doc: ScrapeDocument,
}

async fn scrape(
    State(state): State<AppState>,
    body: Result<Json<ScrapeRequest>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let Json(req) = match body {
        Ok(j) => j,
        Err(rej) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": rej.body_text() })),
            )
        }
    };

    if let Some(db_arc) = &state.db {
        let model = match lock_hist(db_arc) {
            Ok(db) => db.model(&req.doc.model_id),
            Err(err) => return histdb_error(err),
        };
        let model = match model {
            Ok(Some(model)) => model,
            Ok(None) => {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({
                        "error": format!("model {} is not registered", req.doc.model_id)
                    })),
                )
            }
            Err(err) => return histdb_error(err),
        };

        if req.doc.activations.is_empty() {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": "activations must not be empty" })),
            );
        }

        for row in &req.doc.activations {
            if row.layer as i64 >= model.n_layers {
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({
                        "error": format!(
                            "layer {} is outside model {} (n_layers={})",
                            row.layer, model.model_id, model.n_layers
                        )
                    })),
                );
            }
            for expert in &row.expert_ids {
                if *expert as i64 >= model.n_experts {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(serde_json::json!({
                            "error": format!(
                                "expert {} is outside model {} (n_experts={})",
                                expert, model.model_id, model.n_experts
                            )
                        })),
                    );
                }
            }
        }

        match write_scrape(&req.out, &req.doc) {
            Ok(()) => {}
            Err(err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": err.to_string() })),
                )
            }
        }

        let mut db = match lock_hist(db_arc) {
            Ok(db) => db,
            Err(err) => return histdb_error(err),
        };

        let request_ids = match db.request_ids() {
            Ok(ids) => ids,
            Err(err) => return histdb_error(err),
        };
        if request_ids.iter().any(|id| id == &req.doc.request_id) {
            return (
                StatusCode::OK,
                Json(serde_json::json!({ "ingested": 0, "duplicate": true })),
            );
        }

        let record = to_hist_scrape_record(&req.doc);
        match db.insert_scrape(&record) {
            Ok(()) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ingested": req.doc.activations.len(),
                    "duplicate": false,
                })),
            ),
            Err(err) => histdb_error(err),
        }
    } else {
        // Same write_scrape() the CLI's `write-scrape` subcommand calls, so the file
        // on disk is byte-identical whichever front end produced it.
        match write_scrape(&req.out, &req.doc) {
            Ok(()) => (
                StatusCode::OK,
                Json(serde_json::json!({ "written": req.out })),
            ),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": err.to_string() })),
            ),
        }
    }
}

fn lock_hist(db: &Arc<StdMutex<HistDb>>) -> Result<std::sync::MutexGuard<'_, HistDb>, String> {
    db.lock().map_err(|_| "histodb mutex poisoned".to_string())
}

fn histdb_error(err: impl std::fmt::Display) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(serde_json::json!({ "error": err.to_string() })),
    )
}

fn to_hist_scrape_record(doc: &ScrapeDocument) -> ScrapeRecord {
    ScrapeRecord {
        schema_version: SCHEMA_VERSION,
        model_id: doc.model_id.clone(),
        request_id: doc.request_id.clone(),
        class_id: Some(doc.class_id.clone()),
        source: doc.source.clone(),
        phase: doc.phase.clone(),
        n_tokens: Some(doc.n_tokens as i64),
        dtype: Some(doc.dtype.clone()),
        activations: doc
            .activations
            .iter()
            .map(|row| HistActivationRow {
                layer: row.layer as i64,
                token_pos: row.token_pos as i64,
                expert_ids: row.expert_ids.iter().map(|id| *id as i64).collect(),
            })
            .collect(),
        token_ids: None,
    }
}

/// Serve until the process is signalled. Returns only on a bind failure.
pub async fn serve(addr: SocketAddr, state: Shared) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state)).await
}

pub async fn serve_with_router(addr: SocketAddr, router: Router) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await
}
