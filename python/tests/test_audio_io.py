from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from audio_io import WavFormatError, load_wav_pcm16_mono, read_wav_metadata


FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "mono16k_pcm16.wav"


class AudioIOTests(unittest.TestCase):
    def test_load_wav_pcm16_mono_returns_float32_waveform(self):
        waveform = load_wav_pcm16_mono(FIXTURE_WAV)

        self.assertEqual(waveform.dtype, np.float32)
        self.assertGreater(waveform.size, 0)
        self.assertLessEqual(float(np.max(waveform)), 1.0)
        self.assertGreaterEqual(float(np.min(waveform)), -1.0)

    def test_load_wav_pcm16_mono_rejects_non_mono(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_wav = Path(temp_dir) / "stereo.wav"
            with wave.open(str(bad_wav), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(16_000)
                wf.writeframes(b"\x00\x00" * 100)

            with self.assertRaises(WavFormatError):
                load_wav_pcm16_mono(bad_wav)

    def test_load_wav_pcm16_mono_rejects_non_16k(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_wav = Path(temp_dir) / "sr8k.wav"
            with wave.open(str(bad_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8_000)
                wf.writeframes(b"\x00\x00" * 100)

            with self.assertRaises(WavFormatError):
                load_wav_pcm16_mono(bad_wav)

    def test_load_wav_pcm16_mono_rejects_non_16bit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_wav = Path(temp_dir) / "8bit.wav"
            with wave.open(str(bad_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(1)  # 8-bit, not 16-bit
                wf.setframerate(16_000)
                wf.writeframes(b"\x00" * 100)

            with self.assertRaises(WavFormatError):
                load_wav_pcm16_mono(bad_wav)

    def test_load_wav_pcm16_mono_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_wav_pcm16_mono("/nonexistent/path/audio.wav")


class ReadWavMetadataTests(unittest.TestCase):
    def test_read_wav_metadata_returns_correct_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "test.wav"
            n_frames = 160
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16_000)
                wf.writeframes(b"\x00\x00" * n_frames)

            meta = read_wav_metadata(wav_path)

        self.assertEqual(meta.channels, 1)
        self.assertEqual(meta.sample_width, 2)
        self.assertEqual(meta.sample_rate, 16_000)
        self.assertEqual(meta.frame_count, n_frames)

    def test_read_wav_metadata_does_not_require_valid_format(self):
        # read_wav_metadata should return metadata even for non-16kHz or non-mono files
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "stereo44k.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(44_100)
                wf.writeframes(b"\x00\x00" * 200)

            meta = read_wav_metadata(wav_path)

        self.assertEqual(meta.channels, 2)
        self.assertEqual(meta.sample_rate, 44_100)

    def test_read_wav_metadata_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_wav_metadata("/nonexistent/path/audio.wav")


if __name__ == "__main__":
    unittest.main()
