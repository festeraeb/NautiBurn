-- 004: prompt awareness on requests.
--
-- Why: before this, 98.7% of the corpus was prompt-blind. Activations were
-- keyed by request_id, but nothing said which prompt produced the routing, so
-- a held-out split could only be by chunk or by date -- never by prompt -- and
-- the same prompt across two models could not be joined at all. class_id is a
-- seat label, not a prompt: every request from a seat shares one.
--
-- prompt_sha256 is over the prompt TEXT, deliberately not over token ids.
-- Token ids differ per tokenizer, so hashing them splits one prompt into one
-- identity per model and silently destroys exactly the cross-model comparison
-- this corpus exists to support. (Verified 2026-11-04: hashing token ids gave
-- 32 "distinct prompts" for 8 real ones.)
--
-- prompt_text is nullable on purpose. A caller that will not store user text
-- can send the hash alone and still get held-out splits by prompt.
--
-- forge_* are the handoff linkage: they let an activation row be joined back
-- to the forge run and role that caused it, so provenance of a handoff stops
-- needing to be recorded by hand.

ALTER TABLE requests ADD COLUMN prompt_sha256 TEXT;
ALTER TABLE requests ADD COLUMN prompt_text   TEXT;
ALTER TABLE requests ADD COLUMN forge_run_id  TEXT;
ALTER TABLE requests ADD COLUMN forge_role    TEXT;
ALTER TABLE requests ADD COLUMN task_domain   TEXT;

CREATE INDEX IF NOT EXISTS idx_req_prompt ON requests(prompt_sha256);
CREATE INDEX IF NOT EXISTS idx_req_forge  ON requests(forge_run_id, forge_role);
