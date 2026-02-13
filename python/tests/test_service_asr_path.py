from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from service import ASRRequestError, ResidentModelRuntime, create_app

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency in local venv
    TestClient = None


FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "mono16k_pcm16.wav"


class _StubASRModule:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **kwargs: object) -> dict[str, str]:
        self.calls.append((audio, kwargs))
        return {"text": "stub transcript"}


class ServiceASRPathTests(unittest.TestCase):
    def test_transcribe_uses_waveform_when_ffmpeg_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = ResidentModelRuntime(state_dir=Path(temp_dir) / "state")
            runtime._resolve_ffmpeg_path = lambda: (None, "not_found")  # type: ignore[method-assign]
            runtime._refresh_ffmpeg_capability()

            stub = _StubASRModule()
            runtime._asr_module = stub

            text = runtime._transcribe_audio_single(str(FIXTURE_WAV), language="auto")

            self.assertEqual(text, "stub transcript")
            self.assertEqual(len(stub.calls), 1)
            audio_arg = stub.calls[0][0]
            self.assertIsInstance(audio_arg, np.ndarray)
            self.assertEqual(audio_arg.dtype, np.float32)

    def test_transcribe_returns_structured_error_for_invalid_wav_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = ResidentModelRuntime(state_dir=Path(temp_dir) / "state")
            runtime._resolve_ffmpeg_path = lambda: (None, "not_found")  # type: ignore[method-assign]
            runtime._refresh_ffmpeg_capability()
            runtime._asr_module = _StubASRModule()

            invalid_wav = Path(temp_dir) / "invalid.wav"
            with wave.open(str(invalid_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8_000)
                wf.writeframes(b"\x00\x00" * 200)

            with self.assertRaises(ASRRequestError) as ctx:
                runtime._transcribe_audio_single(str(invalid_wav), language="auto")

            self.assertEqual(ctx.exception.error_code, "asr_wav_format_unsupported")
            self.assertIn("未找到可用 ffmpeg", ctx.exception.human_message)

    @unittest.skipIf(TestClient is None, "fastapi testclient requires httpx")
    def test_asr_endpoint_success_with_valid_wav_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = ResidentModelRuntime(state_dir=Path(temp_dir) / "state")
            runtime._resolve_ffmpeg_path = lambda: (None, "not_found")  # type: ignore[method-assign]
            runtime._refresh_ffmpeg_capability()
            runtime._asr_module = _StubASRModule()

            app = create_app(runtime)
            with TestClient(app) as client:
                response = client.post("/asr/transcribe", json={"audio_path": str(FIXTURE_WAV)})

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body.get("text"), "stub transcript")

    @unittest.skipIf(TestClient is None, "fastapi testclient requires httpx")
    def test_asr_endpoint_returns_422_with_error_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = ResidentModelRuntime(state_dir=Path(temp_dir) / "state")
            runtime._resolve_ffmpeg_path = lambda: (None, "not_found")  # type: ignore[method-assign]
            runtime._refresh_ffmpeg_capability()
            runtime._asr_module = _StubASRModule()

            invalid_wav = Path(temp_dir) / "invalid.wav"
            with wave.open(str(invalid_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8_000)
                wf.writeframes(b"\x00\x00" * 200)

            app = create_app(runtime)
            with TestClient(app) as client:
                response = client.post("/asr/transcribe", json={
                    "audio_path": str(invalid_wav),
                    "audio_enhancement_enabled": False,
                })

            self.assertEqual(response.status_code, 422)
            body = response.json()
            self.assertEqual(body.get("error_code"), "asr_wav_format_unsupported")
            self.assertIn("human_message", body)


if __name__ == "__main__":
    unittest.main()
