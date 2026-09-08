import pytest
import subprocess
from pathlib import Path

from src.audio_service import (
    validate_audio_with_ffprobe,
    get_audio_duration,
    normalize_audio,
    split_audio_into_chunks,
    prepare_audio,
)

@pytest.fixture
def valid_audio(tmp_path):
    path = tmp_path / "valid.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=2", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True
    )
    return str(path)

@pytest.fixture
def short_audio(tmp_path):
    path = tmp_path / "short.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=5", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True
    )
    return str(path)

@pytest.fixture
def long_audio(tmp_path):
    path = tmp_path / "long.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=30", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        check=True, capture_output=True
    )
    return str(path)

@pytest.fixture
def empty_file(tmp_path):
    path = tmp_path / "empty.wav"
    path.touch()
    return str(path)

@pytest.fixture
def no_audio_stream_file(tmp_path):
    path = tmp_path / "no_audio.wav"
    path.write_text("This is not an audio file.")
    return str(path)

class TestValidateAudio:
    def test_validate_valid_audio_file(self, valid_audio):
        """Test validation passes for a valid audio file."""
        result = validate_audio_with_ffprobe(valid_audio)
        assert "duration" in result
        assert "metadata" in result
    
    def test_validate_missing_file(self):
        """Test validation fails for missing file."""
        with pytest.raises(FileNotFoundError):
            validate_audio_with_ffprobe("/nonexistent/file.wav")
    
    def test_validate_empty_file(self, empty_file):
        """Test validation fails for empty audio file."""
        with pytest.raises(ValueError, match="Audio file is empty."):
            validate_audio_with_ffprobe(empty_file)
    
    def test_validate_no_audio_stream(self, no_audio_stream_file):
        """Test validation fails if file has no audio stream."""
        with pytest.raises(ValueError):
            validate_audio_with_ffprobe(no_audio_stream_file)

class TestNormalizeAudio:
    def test_normalize_short_audio(self, short_audio, tmp_path):
        """Test normalizing short audio file to 16 kHz WAV."""
        out_path = tmp_path / "normalized.wav"
        res = normalize_audio(short_audio, str(out_path))
        assert res["success"] is True
        assert res["format"] == "wav"
        assert res["sample_rate"] == 16000
        assert res["channels"] == 1
    
    def test_normalize_output_format(self, valid_audio, tmp_path):
        """Test normalized output is WAV at 16 kHz."""
        out_path = tmp_path / "normalized.wav"
        normalize_audio(valid_audio, str(out_path))
        assert out_path.exists()
        info = validate_audio_with_ffprobe(str(out_path))
        # Ensure it is WAV format and 16000 sample rate
        assert "pcm_s16le" in info["metadata"] or "16000" in info["metadata"]
    
    def test_normalize_creates_output_directory(self, valid_audio, tmp_path):
        """Test that output directory is created if missing."""
        out_path = tmp_path / "deeply" / "nested" / "dir" / "out.wav"
        normalize_audio(valid_audio, str(out_path))
        assert out_path.exists()
    
    def test_normalize_missing_input_file(self):
        """Test normalization fails for missing input."""
        with pytest.raises(FileNotFoundError):
            normalize_audio("/nonexistent/input.wav", "/tmp/output.wav")
    
    def test_normalize_empty_input_file(self, empty_file, tmp_path):
        """Test normalization fails for empty input."""
        out_path = tmp_path / "out.wav"
        with pytest.raises(ValueError):
            normalize_audio(empty_file, str(out_path))

class TestChunkAudio:
    def test_chunk_short_audio_no_chunking(self, short_audio, tmp_path):
        """Test short audio (< 10s) generates a single chunk."""
        res = split_audio_into_chunks(short_audio, str(tmp_path), chunk_duration=10)
        assert res["chunk_count"] == 1
        assert len(res["chunks"]) == 1
        assert "chunk_0000.wav" in res["chunks"][0]["path"]
    
    def test_chunk_long_audio_multiple_chunks(self, long_audio, tmp_path):
        """Test long audio (> 10s) is split into chunks."""
        res = split_audio_into_chunks(long_audio, str(tmp_path), chunk_duration=10)
        assert len(res["chunks"]) == 3
        assert res["chunks"][0]["index"] == 0
        assert res["chunks"][0]["duration_ms"] == 10000
        assert res["chunks"][1]["index"] == 1
        assert res["chunks"][1]["duration_ms"] == 10000
        assert res["chunks"][2]["index"] == 2
        assert res["chunks"][2]["duration_ms"] == 10000
    
    def test_chunk_naming_convention(self, long_audio, tmp_path):
        """Test chunks are named chunk_0000.wav, chunk_0001.wav, etc."""
        res = split_audio_into_chunks(long_audio, str(tmp_path), chunk_duration=10)
        assert "chunk_0000.wav" in res["chunks"][0]["path"]
        assert "chunk_0001.wav" in res["chunks"][1]["path"]
        assert "chunk_0002.wav" in res["chunks"][2]["path"]
    
    def test_chunk_metadata_accuracy(self, long_audio, tmp_path):
        """Test chunk metadata (start_ms, end_ms, duration_ms) is accurate."""
        res = split_audio_into_chunks(long_audio, str(tmp_path), chunk_duration=10)
        assert res["chunks"][0]["start_ms"] == 0
        assert res["chunks"][0]["end_ms"] == 10000
        assert res["chunks"][1]["start_ms"] == 10000
        assert res["chunks"][1]["end_ms"] == 20000
    
    def test_chunk_all_files_exist(self, long_audio, tmp_path):
        """Test all chunk files are created and readable."""
        res = split_audio_into_chunks(long_audio, str(tmp_path), chunk_duration=10)
        for chunk in res["chunks"]:
            p = Path(chunk["path"])
            assert p.exists()
            assert p.stat().st_size > 0

class TestPrepareAudio:
    def test_prepare_short_audio_end_to_end(self, short_audio, tmp_path):
        """Test full pipeline: 5-second audio -> normalized + no chunks."""
        res = prepare_audio(
            input_audio_path=short_audio,
            job_id="test_job_short",
            output_base_dir=str(tmp_path)
        )
        assert res["is_chunked"] is False
        assert Path(res["normalized_audio_path"]).exists()
        assert len(res["chunks"]) == 0
        assert res["total_duration_ms"] > 0
    
    def test_prepare_long_audio_end_to_end(self, long_audio, tmp_path):
        """Test full pipeline: 30-second audio -> normalized + 3 chunks."""
        res = prepare_audio(
            input_audio_path=long_audio,
            job_id="test_job_long",
            output_base_dir=str(tmp_path)
        )
        assert res["is_chunked"] is True
        assert Path(res["normalized_audio_path"]).exists()
        assert len(res["chunks"]) == 3
        assert all(Path(c["path"]).exists() for c in res["chunks"])
    
    def test_prepare_creates_job_directory(self, short_audio, tmp_path):
        """Test job-specific directory is created: {base}/{job_id}/audio/"""
        prepare_audio(short_audio, "test_job", str(tmp_path))
        expected_dir = tmp_path / "test_job" / "audio"
        assert expected_dir.exists()
        assert (expected_dir / "normalized_audio.wav").exists()
    
    def test_prepare_output_matches_interface_contract(self, short_audio, tmp_path):
        """Test returned dict matches AUDIO_INTERFACE.md structure."""
        res = prepare_audio(short_audio, "test_job", str(tmp_path))
        assert "is_chunked" in res
        assert "normalized_audio_path" in res
        assert "total_duration_ms" in res
        assert "chunks" in res
        assert isinstance(res["is_chunked"], bool)
        assert isinstance(res["chunks"], list)

class TestErrorHandling:
    def test_prepare_audio_invalid_input(self, tmp_path):
        """Test prepare_audio raises clear error for invalid input."""
        with pytest.raises((FileNotFoundError, ValueError)):
            prepare_audio("/nonexistent/file.wav", "test_job", str(tmp_path))
    
    def test_error_messages_are_descriptive(self, tmp_path):
        """Test error messages are human-readable and helpful."""
        try:
            prepare_audio("/nonexistent/file.wav", "test_job", str(tmp_path))
        except FileNotFoundError as e:
            assert "not found" in str(e).lower()
