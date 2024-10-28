import os
from time import sleep
from dataclasses import dataclass
from typing import Union, Tuple, List
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.python.framework.ops import EagerTensor

from AudioRzeczy.config import Config
from AudioRzeczy.logger import CustomLogger

cl = CustomLogger(
    logger_name=__file__.split("\\")[-1],
    logger_log_level=Config.TERMINAL_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL
)
logger = cl.create_logger()


@dataclass
class RMSCalc:
    sample_rate: int

    def __post_init__(self) -> None:
        logger.debug(f"Initializing AudioPredictor with: \n"
                     f"{self.sample_rate=}")

    @staticmethod
    def calculate_rms(signal: np.ndarray) -> float:
        """Oblicza RMS (Root Mean Square) sygnału."""
        return np.round(np.sqrt(np.mean(signal ** 2)), 6)

    def is_voice_present(self, signal: np.ndarray, rms_threshold: float) -> Tuple[float, bool]:
        rms_value = self.calculate_rms(signal)
        logger.debug(f"{rms_value=}")
        return rms_value, rms_value > rms_threshold

    def get_audio_rms(self, audio: np.ndarray, step: float = .1,
                      rms_threshold_range: Tuple[Union[float, int], Union[float, int]] = (0, 100),
                      filterout: bool = False) -> Tuple[List[float], np.ndarray]:

        chunk_size = int(step * self.sample_rate)
        num_chunks = int(len(audio) / chunk_size)
        selected_chunks = []

        rms_list = []
        for i in range(num_chunks):
            chunk = audio[i * chunk_size:(i + 1) * chunk_size]
            rms_val = self.calculate_rms(signal=chunk)
            if filterout:
                if rms_threshold_range[0] < rms_val < rms_threshold_range[1]:
                    selected_chunks.append(chunk)
                    rms_list.append(rms_val)
            else:
                selected_chunks.append(chunk)
                rms_list.append(rms_val)
        if selected_chunks:
            voice_audio = np.concatenate(selected_chunks, axis=0)
        else:
            voice_audio = np.array([])

        return rms_list, voice_audio

    def record_and_extract_voice(self, duration: int = 2, step: float = .1,
                                 rms_threshold_range: Tuple[Union[float, int], Union[float, int]]
                                 = (0, 100)) -> np.ndarray:
        logger.info(f"Recording sound for {duration} seconds...")

        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype=np.float32)
        sd.wait()
        audio = np.int16(audio * 32767)

        chunk_size = int(step * self.sample_rate)
        num_chunks = int(len(audio) / chunk_size)
        selected_chunks = []

        for i in range(num_chunks):
            chunk = audio[i * chunk_size:(i + 1) * chunk_size]
            rms_val = self.calculate_rms(signal=chunk)
            if rms_threshold_range[0] < rms_val < rms_threshold_range[1]:
                selected_chunks.append(chunk)

        if selected_chunks:
            voice_audio = np.concatenate(selected_chunks, axis=0)
        else:
            voice_audio = np.array([])

        max_len = self.sample_rate
        if len(voice_audio) > max_len:
            voice_audio = voice_audio[:max_len]

        logger.info(f"Extracted {len(voice_audio) / self.sample_rate} seconds of voice audio.")

        return voice_audio


@dataclass
class PreprocessAudio:
    sample_rate: int

    def __post_init__(self) -> None:
        logger.debug(f"Initializing AudioPredictor with: \n"
                     f"{self.sample_rate=}")

    def get_spectrogram(self, waveform: EagerTensor) -> EagerTensor:
        # print(type(waveform))
        # Zero-padding for an audio waveform with less than 16,000 samples.
        sample_rate = self.sample_rate
        waveform = waveform[:sample_rate]
        zero_padding = tf.zeros(
            [self.sample_rate] - tf.shape(waveform),
            dtype=tf.float32)
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatenate the waveform with `zero_padding`, which ensures all audio
        # clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    @staticmethod
    def decode_audio(audio_binary: EagerTensor) -> EagerTensor:
        # Decode WAV-encoded audio files to `float32` tensors, normalized
        # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        # Since all the data is single channel (mono), drop the `channels`
        # axis from the array.
        return tf.squeeze(audio, axis=-1)

    def preprocess_file_data(self, file_path: Union[str, os.PathLike]) -> EagerTensor:
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
        # print(waveform)
        spectrogram = self.get_spectrogram(waveform)
        normalized_spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension if needed

        return normalized_spectrogram

    def preprocess_mic_data(self, waveform: np.array) -> EagerTensor:
        waveform_pcm = tf.squeeze(waveform, axis=-1)
        spectrogram = self.get_spectrogram(waveform=waveform_pcm)
        normalized_spectrogram = tf.expand_dims(spectrogram, axis=0)

        return normalized_spectrogram

    @staticmethod
    def record_audio(filename: Union[str, None] = None, duration: int = 1,
                     fs: int = 16000, play_audio: bool = False) -> np.array:
        logger.info(f"Record sound for  {duration} seconds...")
        sleep(.3)
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()
        if play_audio:
            sd.play(audio, samplerate=fs)
            sd.wait()
        logger.info("Recording has been completed")

        if filename:
            write(filename, fs, audio)
            logger.info(f"Sounds has been saved to: {filename}")
        return audio

