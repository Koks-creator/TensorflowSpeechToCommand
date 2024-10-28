from dataclasses import dataclass
import os
from typing import Union
import turtle
import sounddevice as sd

from AudioRzeczy.audio_predictor import AudioPredictor
from AudioRzeczy.config import Config
from AudioRzeczy.logger import CustomLogger

logger = CustomLogger(
    logger_name=__file__.split("\\")[-1],
    logger_log_level=Config.TERMINAL_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL
).create_logger()


@dataclass
class SimpleGame:
    model_folder: Union[str, os.PathLike]
    sample_rate: int
    record_duration: Union[float, int]
    record_step: Union[float, int]
    forward_val: int = 100
    check_interval: int = 750
    play_audio: bool = False

    def __post_init__(self):
        self.predictor = AudioPredictor(
            model_folder=self.model_folder,
            sample_rate=self.sample_rate,
        )
        self.screen = turtle.getscreen()
        self.tutel = turtle.Turtle()

        size = self.tutel.turtlesize()
        increase = (2 * num for num in size)
        self.tutel.turtlesize(*increase)

        self.tutel.pensize(5)
        self.tutel.pencolor("blue")

        self.label_turtle = turtle.Turtle()
        self.label_turtle.hideturtle()
        self.label_turtle.penup()
        self.label_turtle.goto(-370, 290)

        self.moves = {
            "up": self.go_up,
            "down": self.go_down,
            "left": self.go_left,
            "right": self.go_right
        }

    def display_command(self, command: str):
        self.label_turtle.clear()
        self.label_turtle.write(f"Predicted Command: {command}", font=("Arial", 16, "bold"))

    def go_right(self):
        self.tutel.right(45)

    def go_left(self):
        self.tutel.left(45)

    def go_up(self):
        current = self.tutel.heading()
        if 0 <= current <= 90:
            self.tutel.right(current - 90)
        elif 90 < current < 180:
            self.tutel.right(current - 90)
        elif 315 <= current <= 360:
            self.tutel.left(current - 180)
        elif 270 <= current < 315:
            self.tutel.left(current - 90)
        elif 180 <= current <= 225:
            self.tutel.right(current - 90)

    def go_down(self):
        current = self.tutel.heading()
        if 0 <= current < 90:
            self.tutel.right(180 - current)
        elif 90 <= current < 180:
            self.tutel.right(270 - current)
        elif 315 <= current <= 360:
            self.tutel.left(270 - current)
        elif 270 <= current < 315:
            self.tutel.left(90 - current)
        elif 180 <= current <= 225:
            self.tutel.right(270 - current)

    def move_turtle(self, command: str):
        self.display_command(command)
        if command in list(self.moves.keys()):
            self.moves[command]()
        elif command == "go":
            self.tutel.forward(self.forward_val)

    def check_voice_command(self):
        self.tutel.fillcolor("red")
        voice_audio = self.predictor.rms_calc.record_and_extract_voice(
            duration=self.record_duration,
            step=self.record_step,
            rms_threshold_range=Config.RMS_THRESHOLD_RANGE
        )

        if self.play_audio:
            sd.play(voice_audio, samplerate=Config.SAMPLE_RATE)
            sd.wait()

        rms_value, is_voice = self.predictor.rms_calc.is_voice_present(
            signal=voice_audio, rms_threshold=Config.RMS_THRESHOLD
        )

        self.tutel.fillcolor("blue")

        if is_voice:
            prediction = self.predictor.get_prediction(audio_data=voice_audio)
            logger.info(f"{prediction=}")
            self.move_turtle(prediction)

            if prediction == "stop":
                turtle.bye()
                print("Leaving")
                logger.info("Leaving")
                return
        turtle.ontimer(self.check_voice_command, self.check_interval)

    def run(self):
        turtle.listen()
        turtle.ontimer(self.check_voice_command, self.check_interval)
        turtle.mainloop()


if __name__ == '__main__':
    simple_game = SimpleGame(
        model_folder=fr"{Config.MODELS_PATH}/{Config.MODEL}",
        sample_rate=Config.SAMPLE_RATE,
        record_duration=Config.RECORDING_DURATION,
        record_step=Config.RECORDING_STEP,
        check_interval=Config.GAME_CHECK_INTERVAL,
        forward_val=Config.GAME_FORWARD_VAL,
        play_audio=Config.GAME_PLAY_AUDIO
    )
    simple_game.run()
