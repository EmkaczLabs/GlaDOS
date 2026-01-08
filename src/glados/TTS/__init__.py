"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


# Factory function
def get_speech_synthesizer(
    voice: str = "glados",
    *,
    language: str = "en_us",
    model_path: str | None = None,
    phoneme_path: str | None = None,
    token_to_idx_path: str | None = None,
    idx_to_token_path: str | None = None,
) -> SpeechSynthesizerProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.
    Parameters:
        voice (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() == "glados":
        from ..TTS import tts_glados

        params: dict[str, object] = {"language": language}
        if model_path is not None:
            params["model_path"] = model_path
        if phoneme_path is not None:
            params["phoneme_path"] = phoneme_path
        if token_to_idx_path is not None:
            params["token_to_idx_path"] = token_to_idx_path
        if idx_to_token_path is not None:
            params["idx_to_token_path"] = idx_to_token_path

        return tts_glados.SpeechSynthesizer(**params)

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    if model_path is not None:
        return tts_kokoro.SpeechSynthesizer(model_path=model_path, voice=voice, language=language)
    return tts_kokoro.SpeechSynthesizer(voice=voice, language=language)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer"]
