-- 005: the run envelope — correlate at the time, not afterwards.
--
-- Before this, a prompt and the experts it routed to could not be joined. The
-- engine minted its own request id (nb-server-<ts>-<seq>) knowing nothing of
-- the request above it, so the only correlation available was a timestamp
-- window — lossy under concurrency, and silently wrong rather than absent when
-- it mismatched.
--
-- The envelope inverts that. Lemonade stamps a run_id at the front door where
-- the prompt is visible; every card that processes part of that run appends a
-- hop. Correlation happens while the thing that knows is still running.
--
-- Two deliberate choices:
--
-- 1. A hop stores a REFERENCE and a DIGEST, not the activation bytes. One
--    request produced ~390 route records in the 2026-11-04 export, so inline
--    scrapes would make the handoff the heaviest thing on the wire. The sha256
--    means a scrape read back later can still be proven to be the one that hop
--    produced, which is what keeps "did we correlate this wrong?" answerable.
--
-- 2. Hops are append-only and sequenced. A card that dies leaves a GAP in the
--    sequence rather than a rewritten envelope, so a lost stage is visible as
--    a lost stage instead of looking like it never existed.

CREATE TABLE IF NOT EXISTS run_envelope (
  run_id        TEXT PRIMARY KEY,          -- minted by Lemonade, never re-derived
  prompt_sha256 TEXT,                      -- over the prompt TEXT, see migration 004
  prompt_text   TEXT,                      -- optional: a caller may send the hash alone
  task_class    TEXT,
  forge_role    TEXT,
  model_id      TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  completed_at  TEXT,
  -- open | complete | abandoned. 'open' with no hop for a long time is the
  -- signal that a card took the work and never came back.
  status        TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_env_prompt ON run_envelope(prompt_sha256);
CREATE INDEX IF NOT EXISTS idx_env_status ON run_envelope(status, created_at);

CREATE TABLE IF NOT EXISTS run_hop (
  run_id        TEXT NOT NULL REFERENCES run_envelope(run_id),
  seq           INTEGER NOT NULL,          -- 1-based; a missing number is a lost hop
  card          TEXT,                      -- node:device, e.g. cesarops4om:1
  engine        TEXT,                      -- llama.cpp-nautiburn | strata | lemonade-burn | ...
  request_id    TEXT,                      -- the engine's own id, joins to activations
  scrape_ref    TEXT,                      -- where the rows landed, not the rows
  scrape_rows   INTEGER,
  scrape_sha256 TEXT,                      -- proves a later read is the same scrape
  t_start_ms    INTEGER,
  t_end_ms      INTEGER,

  -- ── scoring and retention ──
  -- score is overlap_k: of the experts PAMP predicted would fire, how many did.
  -- Measured 2026-11-04 the fleet sits at 3.39/8, so divergence-only saves
  -- roughly 40% today, and more as the predictor improves.
  score             REAL,
  -- MANDATORY on any divergence row. A delta cannot be reconstructed without
  -- the predictor that produced it; if the belief model is retrained and this
  -- is null, the row becomes permanently unreadable.
  predictor_version TEXT,
  -- divergence   : only the experts that differed from prediction were kept
  -- full_lowscore: prediction was bad, whole record kept for analysis
  -- full_control : kept regardless of score, sampled BEFORE scoring
  --
  -- The control class is not optional. Keeping full records only when the
  -- prediction was poor makes the archive a library of the predictor's
  -- failures; training on it would teach the model its own mistakes, and
  -- "hit-rate improved" becomes unfalsifiable because the baseline is chosen
  -- by the thing being measured. A small random sample taken before the score
  -- is known is the only honest control.
  retention     TEXT NOT NULL DEFAULT 'divergence',

  PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_hop_request   ON run_hop(request_id);
CREATE INDEX IF NOT EXISTS idx_hop_retention ON run_hop(retention);
CREATE INDEX IF NOT EXISTS idx_hop_score     ON run_hop(score);
