-- Predict latency on the belief row (microseconds). Scrape JSON v1 stays 1;
-- this is database schema 3. Same `beliefs` table as the handoff — not a
-- second schema. The shadow box records how long the predictor took before
-- the gate; overlap still lives on `disagreements.overlap_k`.
ALTER TABLE beliefs ADD COLUMN latency_us INTEGER;
