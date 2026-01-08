from pathlib import Path

from glados.TTS.phonemizer import Phonemizer, ModelConfig


def test_phonemizer_pl_fallback(tmp_path: Path) -> None:
    # Create a model config with non-existent resource paths to force fallback
    cfg = ModelConfig(
        model_path=tmp_path / "missing_phonemizer.onnx",
        phoneme_dict_path=tmp_path / "missing_lang_dict.pkl",
        token_to_idx_path=tmp_path / "missing_token.pkl",
        idx_to_token_path=tmp_path / "missing_idx.pkl",
    )

    p = Phonemizer(config=cfg)

    result = p.convert_to_phonemes(["Cześć"], lang="pl")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    # Even if espeak is not available, fallback should return a string (at least the input)
    assert result[0] is not None