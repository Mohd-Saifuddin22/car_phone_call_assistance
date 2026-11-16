# Car Phone Call Assistance

A React and FastAPI application for customer inquiries, administrator lead management, and AI-powered phone consultations in English or Hinglish.

## Contents

- `backend/`: API, authentication, database access, audio handling, call orchestration, and preference matching.
- `frontend/`: customer form and administrator interface built with React, TypeScript, and Vite.
- `database/app_schema.sql`: operational database structure without customer data.
- `backend/tests/`: automated tests using synthetic fixtures and mocked services.
- `pyproject.toml`: Python dependencies and build configuration.

## Install

Python 3.11 or newer is required. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
cd frontend
npm ci
npm run build
```

## Private runtime setup

This is a deliberately limited source snapshot. It contains no original Git history, credentials, local environment files, customer data, catalogue contents, recordings, generated audio, private conversation prompts, internal documents, or development workspaces.

A working deployment also requires environment variables described by `backend/config.py`, a separately provisioned read-only vehicle catalogue, a private conversation prompt at the path expected by `backend/conversation_driver.py`, and locally generated disclosure audio. Supply those privately before running `python run_backend.py`. No real provider credentials belong in frontend code or this repository.

The original full test suite includes checks that require these omitted runtime resources. The portability test for omitted local troubleshooting scripts is excluded. Run the resource-independent subset with synthetic settings (these are test placeholders, not credentials):

```sh
PYTHONNOUSERSITE=1 python - <<'PY'
import os
import pytest

for name in (
    "DEEPGRAM_API_KEY", "MERCURY_TWO_API_KEY", "GEMINI_API_KEY",
    "SARVAM_API_KEY", "ADMIN_PASSWORD",
):
    os.environ[name] = "test-only-placeholder"
os.environ["ADMIN_EMAIL"] = "admin@example.com"
raise SystemExit(pytest.main([
    "-q",
    "backend/tests/test_contact.py",
    "backend/tests/test_outcome_classifier.py",
    "backend/tests/test_concurrency.py",
    "backend/tests/test_audio_repacketizer.py",
    "backend/tests/test_db_access.py",
    "backend/tests/test_network_guard.py",
    "backend/tests/test_preference_extractor.py",
    "backend/tests/test_blurb_formatter.py",
]))
PY
```
