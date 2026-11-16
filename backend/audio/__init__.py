"""backend/audio — pure audio framing transforms + STT lexicon for the voice pipeline."""

from backend.audio.lexicon import DEEPGRAM_KEYTERMS, DEEPGRAM_KEYTERMS_HINTS
from backend.audio.outbound_sender import parse_mark_name, run_outbound_sender
from backend.audio.repacketizer import (
    DEEPGRAM_CHUNK_BYTES,
    MULAW_FRAME_BYTES,
    MULAW_FRAME_MS,
    InboundRepacketizer,
    OutboundRepacketizer,
)
from backend.audio.sarvam_tts import AsyncSarvamTTSClient, SarvamTTSError

__all__ = [
    "AsyncSarvamTTSClient",
    "DEEPGRAM_CHUNK_BYTES",
    "DEEPGRAM_KEYTERMS",
    "DEEPGRAM_KEYTERMS_HINTS",
    "InboundRepacketizer",
    "MULAW_FRAME_BYTES",
    "MULAW_FRAME_MS",
    "OutboundRepacketizer",
    "SarvamTTSError",
    "parse_mark_name",
    "run_outbound_sender",
]