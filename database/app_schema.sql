-- =====================================================================
-- app.db — operational data for the AI-Powered Car Consultation System
-- =====================================================================
-- Companion to vehicles.db (a separate, read-only reference catalogue used
-- by the Matcher). This file is the single source of truth for the app.db
-- schema; the prose description lives in project_description.md > Data Model.
--
-- The single `admin` row is NOT seeded here — the application seeds it from
-- the ADMIN_EMAIL / ADMIN_PASSWORD environment variables on first boot
-- (password hashed). This file only defines structure.
--
-- IMPORTANT: SQLite enforces ON DELETE CASCADE only when foreign keys are
-- enabled, and that setting is PER-CONNECTION and OFF by default. The
-- application MUST run `PRAGMA foreign_keys = ON;` on every connection.
-- The line below applies it for the duration of this script.
-- Timestamps are stored as ISO-8601 UTC text via datetime('now').
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- admin — the single admin credential (seeded from env on first boot)
-- ---------------------------------------------------------------------
CREATE TABLE admin (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,                       -- bcrypt/argon2, never plaintext
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------
-- sessions — server-side sessions, rolling ~7-day expiry
-- ---------------------------------------------------------------------
CREATE TABLE sessions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_token_hash TEXT    NOT NULL UNIQUE,           -- sha256 of the cookie token (raw token never stored)
    admin_id           INTEGER NOT NULL,
    csrf_token         TEXT    NOT NULL,                  -- per-session synchronizer CSRF; compared to X-CSRF-Token header
    created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    expires_at         TEXT    NOT NULL,                  -- rolling; refreshed forward on each authenticated request
    FOREIGN KEY (admin_id) REFERENCES admin(id) ON DELETE CASCADE
);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);   -- for the periodic expiry sweep

-- ---------------------------------------------------------------------
-- leads — one row per actionable lead (denormalized call-progress state)
-- ---------------------------------------------------------------------
CREATE TABLE leads (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    phone             TEXT    NOT NULL,                   -- canonical E.164 (+91...)
    email             TEXT    NOT NULL,                   -- lowercased/trimmed
    city              TEXT    NOT NULL,                   -- free-text combobox value
    language          TEXT    NOT NULL CHECK (language IN ('English','Hinglish')),
    vehicle_id        INTEGER,                            -- references vehicles.id BY CONVENTION (no cross-DB FK); NULL = "Not specified"
    status            TEXT    NOT NULL DEFAULT 'New'
                              CHECK (status IN ('New','Processing','Completed','Failed')),
    failure_reason    TEXT    CHECK (failure_reason IN ('no_answer','dropped','error')),  -- set only when status='Failed' (app-enforced)
    completed_timeout INTEGER NOT NULL DEFAULT 0 CHECK (completed_timeout IN (0,1)),       -- 1 = Completed via max-duration save
    attempt_count     INTEGER NOT NULL DEFAULT 0,         -- incremented per dial; powers the max-attempts cap
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),   -- submission time
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now'))    -- app sets on each state change
);
CREATE INDEX idx_leads_phone  ON leads(phone);   -- dedupe, block-check, per-phone history
CREATE INDEX idx_leads_email  ON leads(email);
CREATE INDEX idx_leads_status ON leads(status);  -- dashboard filtering / concurrency counting
CREATE UNIQUE INDEX idx_leads_phone_active ON leads(phone) WHERE status IN ('New', 'Processing');  -- prevents duplicate active leads for the same phone

-- ---------------------------------------------------------------------
-- call_attempts — one row per dial attempt (per-attempt call data + history)
-- ---------------------------------------------------------------------
CREATE TABLE call_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id         INTEGER NOT NULL,
    attempt_number  INTEGER NOT NULL,                     -- 1,2,3... matching the lead's sequence
    twilio_call_sid TEXT,                                 -- NULL if create-call failed before a SID was issued
    outcome         TEXT    CHECK (outcome IN ('completed','no_answer','dropped','error')),  -- NULL while the call is live
    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),   -- dial start; drives the live-view elapsed timer
    ended_at        TEXT,                                 -- NULL while live
    recording_path  TEXT,                                 -- local file path; set on Twilio recordingStatusCallback
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE CASCADE
);
CREATE INDEX idx_call_attempts_lead_id    ON call_attempts(lead_id);
CREATE INDEX idx_call_attempts_twilio_sid ON call_attempts(twilio_call_sid);  -- webhooks resolve the attempt by CallSid

-- ---------------------------------------------------------------------
-- transcript_turns — one row per turn, linked to a call attempt
-- ---------------------------------------------------------------------
CREATE TABLE transcript_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    call_attempt_id INTEGER NOT NULL,
    turn_index      INTEGER NOT NULL,                     -- order within the call
    speaker         TEXT    NOT NULL CHECK (speaker IN ('customer','riya')),
    text            TEXT    NOT NULL,                     -- customer = Deepgram STT; riya = Mercury text
    interrupted     INTEGER NOT NULL DEFAULT 0 CHECK (interrupted IN (0,1)),  -- Riya turn cut off by barge-in
    played_ms       INTEGER,                                               -- audio already spoken by riya when customer interrupted (ms); NULL = not yet known
    total_ms        INTEGER,                                               -- total riya audio duration for this turn (ms); NULL = not yet known
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),   -- turn timestamp
    FOREIGN KEY (call_attempt_id) REFERENCES call_attempts(id) ON DELETE CASCADE
);
CREATE INDEX idx_transcript_turns_attempt ON transcript_turns(call_attempt_id, turn_index);

-- ---------------------------------------------------------------------
-- recommendations — final ranked set, one per lead
-- ---------------------------------------------------------------------
CREATE TABLE recommendations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id             INTEGER NOT NULL UNIQUE,          -- at most one set per lead
    source_attempt_id   INTEGER,                          -- which call_attempts row produced it
    items               TEXT    NOT NULL,                 -- JSON: ordered [{vehicle_id, rank, score}, ...]
    blurb_text          TEXT,                             -- Gemini-formatted readout, stored once (never regenerated on view); nullable until first generation
    preference_snapshot TEXT,                             -- JSON: final preference set the ranking was based on
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (lead_id)           REFERENCES leads(id)         ON DELETE CASCADE,
    FOREIGN KEY (source_attempt_id) REFERENCES call_attempts(id) ON DELETE SET NULL
);

-- ---------------------------------------------------------------------
-- block_list — one row per blocked customer (OR-matching on phone/email)
-- ---------------------------------------------------------------------
CREATE TABLE block_list (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    phone      TEXT    NOT NULL UNIQUE,                   -- E.164; UNIQUE => re-blocking the same number is idempotent (also provides the lookup index)
    email      TEXT    NOT NULL,                          -- lowercased/trimmed; NOT unique (families may share an email)
    name       TEXT,                                      -- snapshot at block time, for a readable Blocked Customers list
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_block_list_email ON block_list(email);   -- phone is already indexed via its UNIQUE constraint
