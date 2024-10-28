import os
from typing import Union
import sounddevice as sd
import soundfile as sf
import numpy as np

from AudioRzeczy.audio_tools import RMSCalc, PreprocessAudio
from AudioRzeczy.config import Config

LOCAL_RMS_THRESHOLD_RANGE = (.5, 50)
AUDIO_FOLDER = "./TestAudio"
audio_prep = PreprocessAudio(
    sample_rate=Config.SAMPLE_RATE
)
rms_calc = RMSCalc(
    sample_rate=Config.SAMPLE_RATE
)


def file_rms(file_path: Union[str, os.PathLike], play_audio: bool = False) -> None:
    data, samplerate = sf.read(file_path)
    audio_pcm = np.int16(data * 32767)
    # print(data.shape)
    # print(rms_calc.calculate_rms(audio_pcm))
    rms_list, filtered_audio = rms_calc.get_audio_rms(audio=audio_pcm,
                                                      filterout=True,
                                                      rms_threshold_range=LOCAL_RMS_THRESHOLD_RANGE)
    # print(filtered_audio.shape)
    if play_audio:
        sd.play(filtered_audio, samplerate=samplerate)
        sd.wait()

    print(f"File: {file_path}")
    print(f"{rms_list}")
    print(f"{min(rms_list)=}")
    print(f"{max(rms_list)=}")
    print(f"Extracted {len(filtered_audio) / samplerate} seconds of audio.")
    rms_val = rms_calc.calculate_rms(filtered_audio)
    print(f"{rms_val=}")
    print(rms_calc.is_voice_present(filtered_audio, rms_threshold=25))


if __name__ == '__main__':
    file_rms(fr"{AUDIO_FOLDER}/klawiatura.wav", play_audio=True)
    # audio_prep.record_audio(fr"{AUDIO_FOLDER}/klawiatura.wav", duration=3)