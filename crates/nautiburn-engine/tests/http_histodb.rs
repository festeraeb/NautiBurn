use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Method, Request, Response, StatusCode};
use http_body_util::BodyExt;
use nautiburn_engine::{router_with_histodb, shared, ModelProfile};
use nautiburn_histodb::{HistDb, ModelRecord};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

fn small_model() -> ModelRecord {
    ModelRecord {
        model_id: "test-model".into(),
        arch: "test".into(),
        n_layers: 2,
        n_experts: 8,
        top_k: 2,
        hidden_size: 2048,
        notes: None,
    }
}

fn scrape_body(out: &str, request_id: &str) -> Value {
    json!({
        "out": out,
        "schema_version": 1,
        "model_id": "test-model",
        "request_id": request_id,
        "class_id": "class",
        "source": "test",
        "phase": "decode",
        "n_tokens": 1,
        "dtype": "f16",
        "activations": [
            { "layer": 0, "token_pos": 0, "expert_ids": [0, 1] },
            { "layer": 1, "token_pos": 1, "expert_ids": [2, 3] }
        ],
        "token_ids": [10, 11]
    })
}

async fn post_json(router: &axum::Router, path: &str, body: Value) -> Response<Body> {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    router.clone().oneshot(req).await.unwrap()
}

async fn json_body(resp: Response<Body>) -> Value {
    let bytes = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    serde_json::from_slice(&bytes).unwrap()
}

async fn open_db(path: &std::path::Path) -> HistDb {
    let db = HistDb::open(path).unwrap();
    db.upsert_model(&small_model()).unwrap();
    db
}

#[tokio::test]
async fn ingest_survives_router_drop_and_reopen() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("hist.db");
    let out = dir.path().join("scrape.json").to_str().unwrap().to_string();

    let db = open_db(&db_path).await;
    let router = router_with_histodb(shared(ModelProfile::OLMOE), Some(Arc::new(Mutex::new(db))));

    let resp = post_json(&router, "/scrape", scrape_body(&out, "req-1")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["ingested"], json!(2));
    assert_eq!(body["duplicate"], json!(false));

    drop(router);

    let reopened = HistDb::open(&db_path).unwrap();
    assert_eq!(reopened.request_ids().unwrap(), vec!["req-1".to_string()]);
    // One histogram row per (layer, expert): {0,1} at layer 0 and {2,3} at layer 1.
    assert_eq!(reopened.histogram_row_count().unwrap(), 4);
}

#[tokio::test]
async fn duplicate_request_id_is_not_inserted_twice() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("hist.db");
    let out = dir.path().join("scrape.json").to_str().unwrap().to_string();

    let db = open_db(&db_path).await;
    let router = router_with_histodb(shared(ModelProfile::OLMOE), Some(Arc::new(Mutex::new(db))));

    let first = post_json(&router, "/scrape", scrape_body(&out, "req-2")).await;
    assert_eq!(first.status(), StatusCode::OK);
    let first_body = json_body(first).await;
    assert_eq!(first_body["ingested"], json!(2));
    assert_eq!(first_body["duplicate"], json!(false));

    let second = post_json(&router, "/scrape", scrape_body(&out, "req-2")).await;
    assert_eq!(second.status(), StatusCode::OK);
    let second_body = json_body(second).await;
    assert_eq!(second_body["ingested"], json!(0));
    assert_eq!(second_body["duplicate"], json!(true));

    let db = HistDb::open(&db_path).unwrap();
    assert_eq!(db.request_ids().unwrap(), vec!["req-2".to_string()]);
    assert_eq!(db.histogram_row_count().unwrap(), 4);
    // A second insert would bump the counts to 2, not add rows.
    assert_eq!(db.histogram_count("test-model", "class", "decode", 0, 0).unwrap(), Some(1));
}

#[tokio::test]
async fn unknown_model_is_rejected_before_write() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("hist.db");
    let out = dir.path().join("scrape.json").to_str().unwrap().to_string();

    let db = open_db(&db_path).await;
    let router = router_with_histodb(shared(ModelProfile::OLMOE), Some(Arc::new(Mutex::new(db))));

    let mut body = scrape_body(&out, "req-unknown");
    body["model_id"] = json!("missing-model");

    let resp = post_json(&router, "/scrape", body).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = json_body(resp).await;
    assert!(body["error"].as_str().unwrap().contains("missing-model"));

    let db = HistDb::open(&db_path).unwrap();
    assert_eq!(db.request_ids().unwrap(), Vec::<String>::new());
    assert_eq!(db.histogram_row_count().unwrap(), 0);
}

#[tokio::test]
async fn empty_activations_are_rejected() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("hist.db");
    let out = dir.path().join("scrape.json").to_str().unwrap().to_string();

    let db = open_db(&db_path).await;
    let router = router_with_histodb(shared(ModelProfile::OLMOE), Some(Arc::new(Mutex::new(db))));

    let mut body = scrape_body(&out, "req-empty");
    body["activations"] = json!([]);

    let resp = post_json(&router, "/scrape", body).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = json_body(resp).await;
    assert!(body["error"].as_str().unwrap().contains("activations"));

    let db = HistDb::open(&db_path).unwrap();
    assert_eq!(db.request_ids().unwrap(), Vec::<String>::new());
    assert_eq!(db.histogram_row_count().unwrap(), 0);
}

#[tokio::test]
async fn layer_out_of_range_is_rejected() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("hist.db");
    let out = dir.path().join("scrape.json").to_str().unwrap().to_string();

    let db = open_db(&db_path).await;
    let router = router_with_histodb(shared(ModelProfile::OLMOE), Some(Arc::new(Mutex::new(db))));

    let mut body = scrape_body(&out, "req-layer");
    body["activations"][1]["layer"] = json!(2);

    let resp = post_json(&router, "/scrape", body).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = json_body(resp).await;
    assert!(body["error"].as_str().unwrap().contains("layer"));

    let db = HistDb::open(&db_path).unwrap();
    assert_eq!(db.request_ids().unwrap(), Vec::<String>::new());
    assert_eq!(db.histogram_row_count().unwrap(), 0);
}

#[tokio::test]
async fn expert_id_out_of_range_is_rejected() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("hist.db");
    let out = dir.path().join("scrape.json").to_str().unwrap().to_string();

    let db = open_db(&db_path).await;
    let router = router_with_histodb(shared(ModelProfile::OLMOE), Some(Arc::new(Mutex::new(db))));

    let mut body = scrape_body(&out, "req-expert");
    body["activations"][0]["expert_ids"] = json!([0, 8]);

    let resp = post_json(&router, "/scrape", body).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = json_body(resp).await;
    assert!(body["error"].as_str().unwrap().contains("expert"));

    let db = HistDb::open(&db_path).unwrap();
    assert_eq!(db.request_ids().unwrap(), Vec::<String>::new());
    assert_eq!(db.histogram_row_count().unwrap(), 0);
}
