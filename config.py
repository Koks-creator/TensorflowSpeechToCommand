from pathlib import Path
import logging
from typing import Union, Tuple


class Config:
    SEED: int = 42
    SAMPLE_RATE: int = 16000  # zmienic na samplerate
    RECORDING_DURATION: Union[float, int] = 2
    RECORDING_STEP: Union[float, int] = .1
    RMS_THRESHOLD_RANGE: Tuple[Union[float, int], Union[float, int]] = (1.9, 1000)
    RMS_THRESHOLD: Union[int, float] = 25
    ROOT_PATH: str = Path(__file__).resolve().parent
    MODELS_PATH: str = fr"{ROOT_PATH}/models"
    MODEL: str = "model5"
    TEST_DATA_PATH: str = fr"{ROOT_PATH}/TestData"
    GAME_FORWARD_VAL: int = 100
    GAME_CHECK_INTERVAL: int = 750
    GAME_PLAY_AUDIO: bool = True  # play recorded voice in game
    TERMINAL_LOG_LEVEL: int = logging.INFO
    FILE_LOG_LEVEL: int = logging.INFO

