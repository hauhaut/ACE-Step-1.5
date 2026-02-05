"""
Audio saving and transcoding utility module

Independent audio file operations outside of handler, supporting:
- Save audio tensor/numpy to files (default FLAC format, fast)
- Format conversion (FLAC/WAV/MP3)
- Batch processing
"""

import hashlib
import json
from pathlib import Path
from typing import Union, Optional
import torch
import numpy as np
import torchaudio
from loguru import logger


class AudioSaver:
    """Audio saving and transcoding utility class"""

    def __init__(self, default_format: str = "flac"):
        """
        Initialize audio saver

        Args:
            default_format: Default save format ('flac', 'wav', 'mp3')
        """
        self.default_format = default_format.lower()
        if self.default_format not in ["flac", "wav", "mp3"]:
            logger.warning(f"Unsupported format {default_format}, using 'flac'")
            self.default_format = "flac"

    def save_audio(
        self,
        audio_data: Union[torch.Tensor, np.ndarray],
        output_path: Union[str, Path],
        sample_rate: int = 48000,
        format: Optional[str] = None,
        channels_first: bool = True,
    ) -> str:
        """
        Save audio data to file

        Args:
            audio_data: Audio data, torch.Tensor [channels, samples] or numpy.ndarray
            output_path: Output file path (extension can be omitted)
            sample_rate: Sample rate
            format: Audio format ('flac', 'wav', 'mp3'), defaults to default_format
            channels_first: If True, tensor format is [channels, samples], else [samples, channels]

        Returns:
            Actual saved file path
        """
        format = (format or self.default_format).lower()
        if format not in ["flac", "wav", "mp3"]:
            logger.warning(f"Unsupported format {format}, using {self.default_format}")
            format = self.default_format

        # Ensure output path has correct extension
        output_path = Path(output_path)
        if output_path.suffix.lower() not in ['.flac', '.wav', '.mp3']:
            output_path = output_path.with_suffix(f'.{format}')

        # Convert to torch tensor
        if isinstance(audio_data, np.ndarray):
            if channels_first:
                # numpy [samples, channels] -> tensor [channels, samples]
                audio_tensor = torch.from_numpy(audio_data.T).float()
            else:
                # numpy [samples, channels] -> tensor [samples, channels] -> [channels, samples]
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.dim() == 2 and audio_tensor.shape[0] < audio_tensor.shape[1]:
                    audio_tensor = audio_tensor.T
        else:
            # torch tensor
            audio_tensor = audio_data.cpu().float()
            if not channels_first and audio_tensor.dim() == 2:
                # [samples, channels] -> [channels, samples]
                if audio_tensor.shape[0] > audio_tensor.shape[1]:
                    audio_tensor = audio_tensor.T

        # Ensure memory is contiguous
        audio_tensor = audio_tensor.contiguous()

        # Select backend and save
        try:
            if format == "mp3":
                # MP3 uses ffmpeg backend
                torchaudio.save(
                    str(output_path),
                    audio_tensor,
                    sample_rate,
                    channels_first=True,
                    backend='ffmpeg',
                )
            elif format in ["flac", "wav"]:
                # FLAC and WAV use soundfile backend (fastest)
                torchaudio.save(
                    str(output_path),
                    audio_tensor,
                    sample_rate,
                    channels_first=True,
                    backend='soundfile',
                )
            else:
                # Other formats use default backend
                torchaudio.save(
                    str(output_path),
                    audio_tensor,
                    sample_rate,
                    channels_first=True,
                )

            logger.debug(f"[AudioSaver] Saved audio to {output_path} ({format}, {sample_rate}Hz)")
            return str(output_path)

        except Exception as e:
            try:
                import soundfile as sf
                audio_np = audio_tensor.transpose(0, 1).numpy()  # -> [samples, channels]
                sf.write(str(output_path), audio_np, sample_rate, format=format.upper())
                logger.debug(f"[AudioSaver] Fallback soundfile Saved audio to {output_path} ({format}, {sample_rate}Hz)")
                return str(output_path)
            except Exception as e:
                logger.error(f"[AudioSaver] Failed to save audio: {e}")
                raise


def generate_uuid_from_params(params_dict) -> str:
    """
    Generate deterministic UUID from generation parameters.
    Same parameters will always generate the same UUID.

    Args:
        params_dict: Dictionary of parameters

    Returns:
        UUID string
    """
    params_json = json.dumps(params_dict, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha256(params_json.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    uuid_str = f"{hash_hex[0:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str


# Global default instance
_default_saver = AudioSaver(default_format="flac")


def save_audio(
    audio_data: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    sample_rate: int = 48000,
    format: Optional[str] = None,
    channels_first: bool = True,
) -> str:
    """
    Convenience function: save audio (using default configuration)

    Args:
        audio_data: Audio data
        output_path: Output path
        sample_rate: Sample rate
        format: Format (default flac)
        channels_first: Tensor format flag

    Returns:
        Saved file path
    """
    return _default_saver.save_audio(
        audio_data, output_path, sample_rate, format, channels_first
    )
