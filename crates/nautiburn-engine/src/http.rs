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

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;

use crate::scrape::{write_scrape, ScrapeDocument};
use crate::{Engine, ExpertPass, ModelProfile};

/// Shared by every handler. The CLI builds its own `Engine` per invocation, so
/// there is exactly one instance here for the server's lifetime.
pub type Shared = Arc<Mutex<Engine>>;

pub fn shared(profile: ModelProfile) -> Shared {
    Arc::new(Mutex::new(Engine::new(profile)))
}

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/route", post(route))
        .route("/scrape", post(scrape))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn stats(State(state): State<Shared>) -> impl IntoResponse {
    let engine = state.lock();
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
    State(state): State<Shared>,
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
    let engine = state.lock();
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
    State(_state): State<Shared>,
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

/// Serve until the process is signalled. Returns only on a bind failure.
pub async fn serve(addr: SocketAddr, state: Shared) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state)).await
}
