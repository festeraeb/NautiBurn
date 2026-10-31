# NautiBurn

Nauti Moe local repository. GitHub remote: https://github.com/festeraeb/NautiBurn.git

## Shadow a scrape (pre-prediction + job-to-device log)

Predictors live in `nautiburn-shadow`. They do not change tokens and do not start models.

```bash
# Name L+1 / L+2 from the kept ensemble, compare to the scrape, write misses back.
# Optional --also-decode adds a frozen-prefill → decode naming pass plus a
# combined causal table. --belief-out writes the named set for schedule.
CARGO_TARGET_DIR=/data/cargo-target/nautiburn-b0-main cargo run -p nautiburn-shadow -- \
  prepredict --db histodb.sqlite --scrape scrape.json \
  --also-decode scrape-decode.json --label fp16 --belief-out belief.json

# Log a device that already holds those experts. Does not dispatch or move weights.
CARGO_TARGET_DIR=/data/cargo-target/nautiburn-b0-main cargo run -p nautiburn-shadow -- \
  schedule --belief belief.json --devices crates/nautiburn-shadow/fixtures/devices.stub.json
```

`schedule --devices stub` loads the shipped stub map (example layout, not a placement lock). Live Tiel / Mixtral / coder boxes are marked busy so they are skipped. CPU is the OpenVINO miss-path stub.

Next-expert score (beside the gate, no histogram bump) is still:

```bash
nautiburn-shadow score --db histodb.sqlite --scrape scrape.json --also-decode scrape-decode.json
```

Flags for `prepredict`: `--db`, `--scrape`, `--also-decode`, `--label`, `--belief-out`.
Flags for `schedule`: `--belief`, `--devices`, `--request-id`.
See `crates/nautiburn-shadow/README.md`.
