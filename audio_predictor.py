from dataclasses import dataclass
from glob import glob
import os
from typing import Union
import numpy as np
import tensorflow as tf
import sounddevice as sd

from AudioRzeczy.config import Config
from AudioRzeczy.audio_tools import PreprocessAudio, RMSCalc
from AudioRzeczy.logger import CustomLogger

tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

cl = CustomLogger(
    logger_name=__file__.split("\\")[-1],
    logger_log_level=Config.TERMINAL_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL
)
logger = cl.create_logger()


@dataclass
class AudioPredictor:
    model_folder: Union[str, os.PathLike]
    sample_rate: int

    def __post_init__(self) -> None:
        logger.debug(f"Initializing AudioPredictor with: \n"
                     f"{self.model_folder=} \n"
                     f"{self.sample_rate=} \n"
                     )

        self.audio_preprocess = PreprocessAudio(sample_rate=self.sample_rate)
        self.rms_calc = RMSCalc(sample_rate=self.sample_rate)

        self.model_path = glob(rf"{self.model_folder}/*.h5")
        if len(self.model_path) > 1:
            logger.error(f"Why do you have multiple models here {self.model_folder}?")
            raise Exception(f"Why do you have multiple models here {self.model_folder}?")
        self.model_path = self.model_path[0]

        self.classes_path = glob(rf"{self.model_folder}/*.txt")
        if len(self.classes_path) > 1:
            logger.error(f"Why do you have multiple classes files here {self.model_folder}?")
            raise Exception(f"Why do you have multiple classes files here {self.model_folder}?")
        self.classes_path = self.classes_path[0]

        with open(self.classes_path) as cf:
            self.commands = cf.read().split("\n")
            logger.debug(f"{self.commands=}")

        self.model = tf.keras.models.load_model(self.model_path)
        logger.debug("Model loaded")

    def get_prediction(self, audio_data: Union[str, os.PathLike, np.array]) -> str:
        if isinstance(audio_data, str) or isinstance(audio_data, os.PathLike):
            normalized_spectrogram = self.audio_preprocess.preprocess_file_data(file_path=audio_data)
        else:
            normalized_spectrogram = self.audio_preprocess.preprocess_mic_data(waveform=audio_data)
        prediction = self.model(normalized_spectrogram)
        # print(prediction)
        classid = np.argmax(prediction)

        return self.commands[classid]


if __name__ == '__main__':
    pred = AudioPredictor(
        model_folder="models/model5",
        sample_rate=Config.SAMPLE_RATE,
    )
    print(pred.get_prediction(audio_data=rf"{Config.TEST_DATA_PATH}/left1.wav"))
    while True:
        print(f"Recording for {Config.RECORDING_DURATION}")
        voice_audio = pred.rms_calc.record_and_extract_voice(
            duration=Config.RECORDING_DURATION,
            step=Config.RECORDING_STEP,
            rms_threshold_range=Config.RMS_THRESHOLD_RANGE
        )
        sd.play(voice_audio, samplerate=Config.SAMPLE_RATE)
        sd.wait()
        rms_value, is_voice = pred.rms_calc.is_voice_present(signal=voice_audio, rms_threshold=Config.RMS_THRESHOLD)
        print(f"{is_voice=}")
        print(rms_value)
        if is_voice:
            prediction = pred.get_prediction(audio_data=voice_audio)
            print(prediction)
            # if prediction == "stop":
            #     break
        input("Next?")
