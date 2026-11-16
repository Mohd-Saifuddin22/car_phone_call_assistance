"""
backend.scripts.generate_disclosures — Pre-render the call-start disclosure audio.

Generates two static μ-law files (English + Hinglish) by calling Sarvam's
REST TTS once per language, stripping the RIFF/WAV container Sarvam
returns, and writing pure raw 8 kHz μ-law bytes to:

    backend/audio/assets/disclosure_en.ulaw
    backend/audio/assets/disclosure_hi.ulaw

The voice pipeline plays these as soon as a call connects — the legal
recording disclosure must be instant and outage-resilient, so we never
synthesize it live. This script is the one-shot generator that runs
whenever the disclosure text, voice, or language needs to change.

Usage
─────
    # default — both files, speaker from settings.sarvam_tts_speaker
    python -m backend.scripts.generate_disclosures

    # one language only
    python -m backend.scripts.generate_disclosures --only hi

    # A/B a different voice without touching settings
    python -m backend.scripts.generate_disclosures --speaker kavitha

    # write to a staging directory for review
    python -m backend.scripts.generate_disclosures --out-dir ./out

Required env (loaded via backend.config.Settings):
    SARVAM_API_KEY    — required, refuses to run if blank

Output is byte-accurate: the on-disk files contain only raw μ-law
samples, so the voice pipeline can hand them straight to the outbound
sender without re-framing.

Why a separate script (not a runtime capability)
────────────────────────────────────────────────
Live disclosure synthesis would add a network round-trip + Sarvam TTFB
to every call's connect path, blocking the legal disclosure by ~hundreds
of ms and creating an outage dependency on Sarvam's REST API at the
worst possible moment (call connect). Pre-rendering trades a one-time
operator cost for guaranteed-zero-latency, always-available playback.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import sys
from pathlib import Path

# Allow this script to be run directly as
# ``python backend/scripts/generate_disclosures.py`` without ``pip install -e .``
# having been run. The ``backend`` package is the project root's child, so we
# add the project root (parent of ``backend``) to sys.path if it is missing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx

from backend.config import load_settings

__all__ = [
    "DISCLOSURES",
    "strip_wav_header",
    "synthesize_ulaw",
    "ulaw_duration_seconds",
    "generate_all",
    "main",
]

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

#: Sarvam REST endpoint. Documented at
#: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/rest-api.md
SARVAM_TTS_URL: str = "https://api.sarvam.ai/text-to-speech"

#: Model that handles the mixed Devanagari + Latin script our Hinglish
#: consultation depends on. ``bulbul:v3`` is the only model that
#: preprocesses lakh/EMI/Indian-numeral tokens naturally.
SARVAM_MODEL: str = "bulbul:v3"

#: 8 kHz matches the end-to-end μ-law mono contract used by Twilio
#: Media Streams, Deepgram Flux, and the existing ``AsyncSarvamTTSClient``.
SARVAM_SAMPLE_RATE: int = 8000

#: Requested container/codec. Sarvam wraps the μ-law samples in a 44-byte
#: RIFF/WAVE container; ``strip_wav_header`` removes it before writing.
SARVAM_OUTPUT_CODEC: str = "mulaw"

#: Mixed Hindi/English preprocessing (lakh, EMI, dates). Required for
#: natural Hinglish; benign for English-only calls.
SARVAM_PREPROCESSING: bool = True

#: HTTP timeout for one Sarvam POST. Generous — short utterances should
#: return in <2 s, but we tolerate 30 s of network jitter.
HTTP_TIMEOUT_S: float = 30.0


# ─── Disclosure source-of-truth ───────────────────────────────────────────────

#: Single source of truth for the call-start legal recording disclosure.
#: Edit here ONLY — both files regenerate from this dict on every run.
#: Keys are the language codes used by ``--only``; values carry the
#: Sarvam BCP-47 code, the verbatim text, and the on-disk filename.
DISCLOSURES: dict[str, dict[str, str]] = {
    "en": {
        "language_code": "en-IN",
        "text": (
            "Hi, you're connected with a-b-c car dealers. "
            "Please note, this call is recorded for quality and "
            "training purposes. Let's get started."
        ),
        "filename": "disclosure_en.ulaw",
    },
    "hi": {
        "language_code": "hi-IN",
        "text": (
            "नमस्ते, आप a-b-c car dealers से connected हैं। "
            "कृपया ध्यान दें, यह call quality और training के लिए "
            "record की जा रही है। चलिए शुरू करते हैं।"
        ),
        "filename": "disclosure_hi.ulaw",
    },
}


#: Default output directory: ``<repo>/backend/audio/assets/``. Operator
#: can override with ``--out-dir`` for staging or review.
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "audio" / "assets"


#: μ-law codec code from the WAVEFORMATEX spec (the ``AudioFormat`` field
#: of the RIFF ``fmt `` sub-chunk). Sarvam is required to emit this when
#: ``output_audio_codec="mulaw"`` is requested. Anything else (typically
#: 1 = PCM) means Sarvam ignored the request and the bytes would be
#: unplayable as μ-law — refusing to write them silently prevents
#: shipping static to customers.
#: Ref: https://learn.microsoft.com/en-us/windows/win32/api/mmeapi/ns-mmeapi-waveformatex
_WAVE_FORMAT_MULAW: int = 7


# ─── Pure helpers ─────────────────────────────────────────────────────────────


def strip_wav_header(raw: bytes) -> bytes:
    """
    Strip a RIFF/WAVE container header if present; return pure raw μ-law samples.

    Sarvam wraps the requested codec (μ-law here) in a standard WAV
    container: ``"RIFF" <size> "WAVE" "fmt " <fmt-chunk> "data" <size> <samples>``.
    The voice pipeline consumes pure raw μ-law bytes — anything else
    breaks Deepgram's acoustic model and Twilio's 20 ms frame math.

    Implementation walks RIFF sub-chunks rather than assuming a fixed
    44-byte offset. Some encoders insert ``LIST``/``INFO``/``JUNK`` chunks
    between ``WAVE`` and ``data`` that shift the data position; walking
    the chunk table stays correct under any such insertion.

    Format guard
    ────────────
    The ``fmt `` sub-chunk's ``AudioFormat`` field MUST be
    ``WAVE_FORMAT_MULAW (7)``. If Sarvam ever ignored
    ``output_audio_codec="mulaw"`` and returned PCM (1) instead, the bytes
    would be unplayable as μ-law — Deepgram would hear distorted audio
    and Twilio would emit static. We refuse such payloads loudly so the
    problem surfaces during generation, not during a live customer call.

    Defensive: if the input is not RIFF/WAVE (e.g. Sarvam one day
    returns header-less μ-law), it is returned unchanged so the on-disk
    file is always pure μ-law whether or not a header was present.
    """
    # Minimum RIFF/WAVE header is 12 bytes (no sub-chunks).
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return raw

    pos = 12  # past "RIFF<size>WAVE"
    fmt_format_code: int | None = None
    data_payload: bytes | None = None

    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        # Chunk sizes are little-endian uint32; a malformed size would
        # loop forever, so cap ``pos`` at the input length defensively.
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little")

        if chunk_id == b"fmt ":
            # Standard fmt chunk payload is 16 bytes for PCM/μ-law; the
            # AudioFormat field sits at offset 0 within that payload
            # (i.e. right after the 8-byte chunk tag+size). Require at
            # least 2 bytes of payload — anything less is malformed.
            if chunk_size < 2:
                raise ValueError(
                    f"strip_wav_header: 'fmt ' chunk too small to carry "
                    f"AudioFormat (got {chunk_size}-byte payload)"
                )
            fmt_format_code = int.from_bytes(raw[pos + 8 : pos + 10], "little")

        elif chunk_id == b"data":
            # Capture the data payload now; the μ-law format check runs
            # after the loop so a ``fmt `` chunk that appears after
            # ``data`` (legal in WAV) still gets validated.
            data_payload = raw[pos + 8 : pos + 8 + chunk_size]
            # Don't break — keep walking in case more chunks follow
            # (RIFF permits trailing chunks; we just ignore them).

        # RIFF chunks are word-aligned; advance past tag+size+payload.
        pos += 8 + chunk_size

    if data_payload is None:
        raise ValueError(
            f"strip_wav_header: RIFF/WAVE header detected but no 'data' "
            f"sub-chunk found in {len(raw)} bytes — refusing to truncate blindly"
        )

    if fmt_format_code != _WAVE_FORMAT_MULAW:
        raise ValueError(
            f"strip_wav_header: WAV AudioFormat={fmt_format_code}, expected "
            f"{_WAVE_FORMAT_MULAW} (WAVE_FORMAT_MULAW). Sarvam may have ignored "
            f"output_audio_codec='mulaw' — refusing to write non-μ-law audio "
            f"to a .ulaw file (it would play as static)."
        )

    return data_payload


def ulaw_duration_seconds(byte_count: int, sample_rate: int = SARVAM_SAMPLE_RATE) -> float:
    """
    Return playback duration for a byte count of μ-law at the given sample rate.

    μ-law is 1 byte per sample → duration = byte_count / sample_rate.
    Exposed for the operator summary at the end of a successful run.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return byte_count / sample_rate


# ─── Network call ─────────────────────────────────────────────────────────────


async def synthesize_ulaw(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    text: str,
    language_code: str,
    speaker: str,
) -> bytes:
    """
    Call Sarvam REST once and return pure raw μ-law bytes (header stripped).

    Parameters
    ----------
    client
        A shared ``httpx.AsyncClient``. Reused across both languages so
        we pay TCP+TLS setup once.
    api_key
        Resolved ``SARVAM_API_KEY`` value (operator's secret).
    text
        Verbatim disclosure text. Must be ≤2 500 chars (Sarvam v3 limit)
        — current disclosures are well under 200.
    language_code
        BCP-47 code; ``en-IN`` or ``hi-IN`` are the supported values.
    speaker
        Sarvam speaker voice; lowercase ``priya`` matches the SDK's
        accepted casing (HTTP 422 otherwise).

    Returns
    -------
    bytes
        Pure raw μ-law samples, header stripped. Length is the duration
        in seconds × 8 000.

    Raises
    ------
    httpx.HTTPStatusError
        Non-2xx from Sarvam (invalid credentials, oversize text, …).
    ValueError
        Empty ``audios`` array, malformed base64, missing ``data`` chunk
        in a present RIFF/WAVE header, or any other protocol violation.
    """
    if not text:
        raise ValueError("synthesize_ulaw: text must be non-empty")

    payload = {
        "text": text,
        "target_language_code": language_code,
        "speaker": speaker,
        "model": SARVAM_MODEL,
        "speech_sample_rate": SARVAM_SAMPLE_RATE,
        "output_audio_codec": SARVAM_OUTPUT_CODEC,
        "enable_preprocessing": SARVAM_PREPROCESSING,
    }
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    response = await client.post(
        SARVAM_TTS_URL,
        json=payload,
        headers=headers,
        timeout=HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    body = response.json()

    audios = body.get("audios")
    if not audios or not isinstance(audios, list):
        raise ValueError(
            f"synthesize_ulaw: Sarvam returned no 'audios' array: {body!r}"
        )

    # The REST API returns one base64 segment per input text. We send
    # one text per request, so the array is length 1 in practice — but
    # concatenate defensively so a future multi-segment response works
    # without code changes.
    audio_b64 = "".join(audios)
    audio_bytes = base64.b64decode(audio_b64, validate=True)

    stripped = strip_wav_header(audio_bytes)
    if len(stripped) != len(audio_bytes):
        log.debug(
            "strip_wav_header: removed %d-byte WAV container; "
            "%d raw μ-law bytes remain",
            len(audio_bytes) - len(stripped),
            len(stripped),
        )
    return stripped


# ─── Orchestration ────────────────────────────────────────────────────────────


async def generate_all(
    *,
    api_key: str,
    speaker: str,
    output_dir: Path,
    languages: list[str],
) -> dict[str, Path]:
    """
    Generate every requested disclosure; return ``{lang_key: output_path}``.

    Writes happen after each Sarvam call so a mid-run failure still
    leaves the earlier file(s) on disk for the operator to use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Single shared client — HTTP/2 keep-alive + connection reuse across
    # the two back-to-back Sarvam calls.
    async with httpx.AsyncClient() as client:
        for lang_key in languages:
            cfg = DISCLOSURES[lang_key]
            log.info(
                "Generating %s (%s, %d chars)…",
                cfg["filename"],
                cfg["language_code"],
                len(cfg["text"]),
            )

            ulaw_bytes = await synthesize_ulaw(
                client,
                api_key=api_key,
                text=cfg["text"],
                language_code=cfg["language_code"],
                speaker=speaker,
            )

            out_path = output_dir / cfg["filename"]
            out_path.write_bytes(ulaw_bytes)
            log.info(
                "  → %s  (%d bytes, ~%.2fs @ %d Hz)",
                out_path,
                len(ulaw_bytes),
                ulaw_duration_seconds(len(ulaw_bytes)),
                SARVAM_SAMPLE_RATE,
            )
            written[lang_key] = out_path

    return written


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_only(raw: str) -> list[str]:
    """
    argparse ``type=`` callable: ``--only {en|hi|both}`` → ``list[str]``.

    Lives outside ``main()`` so it can be unit-tested without spinning up
    argparse state. Accepts the canonical values plus ``all`` as an alias
    for ``both``; an empty string is treated as ``both`` for convenience
    with ``--only=``.
    """
    normalized = raw.strip().lower()
    if normalized in ("en", "hi"):
        return [normalized]
    if normalized in ("both", "all", ""):
        return ["en", "hi"]
    raise argparse.ArgumentTypeError(
        f"--only must be 'en', 'hi', or 'both' (got {raw!r})"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser. Kept separate from ``main()`` for testability."""
    parser = argparse.ArgumentParser(
        prog="generate_disclosures",
        description=(
            "Pre-render the call-start legal recording disclosure in μ-law "
            "8 kHz mono using Sarvam Bulbul v3, stripping any RIFF/WAVE "
            "container so the on-disk files are pure raw μ-law samples."
        ),
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help=(
            "Override the Sarvam speaker voice for this run only "
            "(default: settings.sarvam_tts_speaker, typically 'priya')"
        ),
    )
    parser.add_argument(
        "--only",
        type=_parse_only,
        default=["en", "hi"],
        help="Generate only 'en', only 'hi', or 'both' (default: both)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging (Sarvam response, header-strip sizes, etc.)",
    )
    return parser


def _ensure_utf8_stdout() -> None:
    """
    Force ``sys.stdout`` / ``sys.stderr`` to UTF-8 with ``errors='replace'``.

    Windows legacy consoles (cp1252) crash on the μ in this script's
    argparse help text and on Devanagari in the Hinglish disclosure.
    Reconfiguring at startup keeps both the help text and the runtime
    log output readable on any Windows terminal without changing the
    script's source to ASCII-only.

    Python 3.7+ ``TextIOWrapper.reconfigure`` is the supported way; we
    swallow ``AttributeError`` for the rare stdio replacement case.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover — defensive
            pass


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point. Returns a process exit code (``0`` on success).

    No try/except wrapper: a Sarvam or HTTP failure should propagate as a
    non-zero exit + a clear traceback so the operator notices immediately.
    """
    _ensure_utf8_stdout()
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log.debug("CLI args: %s", args)

    settings = load_settings()
    api_key = settings.sarvam_api_key.get_secret_value()
    speaker = args.speaker or settings.sarvam_tts_speaker

    log.info("Sarvam REST endpoint: %s", SARVAM_TTS_URL)
    log.info("Model: %s   codec: %s   rate: %d Hz", SARVAM_MODEL, SARVAM_OUTPUT_CODEC, SARVAM_SAMPLE_RATE)
    log.info("Speaker: %s", speaker)
    log.info("Output directory: %s", args.out_dir)
    log.info("Languages: %s", args.only)

    written = asyncio.run(
        generate_all(
            api_key=api_key,
            speaker=speaker,
            output_dir=args.out_dir,
            languages=args.only,
        )
    )

    log.info("Done. Generated %d file(s):", len(written))
    for lang_key in ("en", "hi"):  # stable display order
        if lang_key not in written:
            continue
        path = written[lang_key]
        size_bytes = path.stat().st_size
        log.info(
            "  %s  %s  (~%.2fs)",
            lang_key,
            path,
            ulaw_duration_seconds(size_bytes),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())