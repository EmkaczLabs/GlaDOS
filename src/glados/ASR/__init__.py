"""ASR processing components."""

from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray

from .mel_spectrogram import MelSpectrogramCalculator


class TranscriberProtocol(Protocol):
    def __init__(self, model_path: str, *args: str, **kwargs: dict[str, str]) -> None: ...
    def transcribe(self, audio_source: NDArray[Any]) -> str: ...
    def transcribe_file(self, audio_path: Path) -> str: ...


# Factory function
def get_audio_transcriber(
    engine_type: str = "ctc", **kwargs: dict[str, Any]
) -> TranscriberProtocol:  # Return type is now a Union of concrete types
    """
    Factory function to get an instance of an audio transcriber based on the specified engine type.

    Parameters:
        engine_type (str): The type of ASR engine to use:
            - "ctc": Connectionist Temporal Classification model (faster, good accuracy)
            - "tdt": Token and Duration Transducer model (best accuracy, slightly slower)
        **kwargs: Additional keyword arguments to pass to the transcriber constructor

    Returns:
        TranscriberProtocol: An instance of the requested audio transcriber

    Raises:
        ValueError: If the specified engine type is not supported
    """
    if engine_type.lower() == "ctc":
        from .ctc_asr import AudioTranscriber as CTCTranscriber

        # Allow overriding model/config paths and other engine-specific kwargs
        params: dict[str, object] = {}
        model_path = kwargs.get("model_path")
        config_path = kwargs.get("config_path")
        if model_path is not None:
            params["model_path"] = model_path
        if config_path is not None:
            params["config_path"] = config_path

        return CTCTranscriber(**params)
    elif engine_type.lower() == "tdt":
        from .tdt_asr import AudioTranscriber as TDTTranscriber

        params = {}
        config_path = kwargs.get("config_path")
        if config_path is not None:
            params["config_path"] = config_path
        return TDTTranscriber(**params)
    else:
        raise ValueError(f"Unsupported ASR engine type: {engine_type}")


__all__ = ["MelSpectrogramCalculator", "TranscriberProtocol", "get_audio_transcriber"]
