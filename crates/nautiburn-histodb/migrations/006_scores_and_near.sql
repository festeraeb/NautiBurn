-- 006: scrape JSON v2 — router scores, expert weights, and runners-up.
--
-- Why: to preload cards we need the experts that NEARLY fired, not only the ones that did
-- (operator, 2026-11-11). A v2 scrape row carries, beside the top-k ids:
--   weights      each fired expert's weight in the expert sum (after the model's normalisation)
--   scores       each fired expert's raw router logit (before softmax / sigmoid)
--   near_ids     runners-up, ranks k+1..2k by the router's selection score
--   near_scores  their raw router logits
--
-- Additive only: two nullable columns and one new table. Existing rows are not rewritten
-- (v1 rows read NULL), and SQLite's ADD COLUMN is a schema-only change, so this is instant
-- even on the 100+ GB live table and safe beside concurrent readers.
--
-- Runners-up go to their OWN table, numbered with rank >= top_k (rank k is the best expert
-- that did not fire). They are NOT written to `activations`: every existing reader of
-- `activations` and the `histogram_counts` aggregate treats a row as "this expert fired",
-- and would silently count near misses as hits. A query that wants both does
-- `SELECT ... FROM activations UNION ALL SELECT ... FROM activation_near`.
--
-- Applied statement by statement; a duplicate column (added by hand before this ran) is
-- skipped, as in 004.

ALTER TABLE activations ADD COLUMN score REAL;   -- raw router logit of the fired expert
ALTER TABLE activations ADD COLUMN weight REAL;  -- its weight in the expert sum

CREATE TABLE IF NOT EXISTS activation_near (
  request_id TEXT NOT NULL REFERENCES requests(request_id),
  phase      TEXT NOT NULL,
  layer      INTEGER NOT NULL,
  token_pos  INTEGER NOT NULL,
  rank       INTEGER NOT NULL,                 -- top_k .. 2*top_k-1
  expert_id  INTEGER NOT NULL,
  score      REAL,                             -- raw router logit
  PRIMARY KEY (request_id, phase, layer, token_pos, rank)
);
