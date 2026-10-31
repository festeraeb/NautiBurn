-- Persist scrape dtype (e.g. f16) on the request. Scrape JSON v1 stays 1;
-- this is database schema 2. Handoff scrape posts dtype next to n_tokens.
ALTER TABLE requests ADD COLUMN dtype TEXT;
