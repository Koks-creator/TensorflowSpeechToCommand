# Tensorflow Speech To Command

## 1. Overview
This repository contains a set of Python scripts designed to recognize voice commands and interact with a simple Turtle-based game using audio inputs. It consists of audio processing tools, model prediction logic, configuration settings, and a logger. The primary functionality includes detecting specific audio commands and translating them into movements within a Turtle graphics environment. 

[![video](https://img.youtube.com/vi/C29EZqQveG0/0.jpg)](https://www.youtube.com/watch?v=C29EZqQveG0)]

## 2. Cleaning audio from noise using RMS
Recorded audio has 2 seconds of length but model takes 1 second max, having only 1 second to say something was not working properly, so I wanted to give user more time to say command but I also wanted to keep this 1 second max audio, so the goal was to extract only (or mostly :P) voice from audio. To accomplish that I used
RMS (Root Mean Square) to filter out noise from audio, this approach allows us to evaluate the energy level of a signal, helping to determine whether a segment of audio contains sound (such as speech) or is simply silence or low-intensity noise. Practically, this means that after calculating the RMS for short audio segments, we can compare these values to a set threshold (RMS threshold) to decide whether the segment contains speech or just background noise.

## 3. Setup
 - Install Python 3.9+
 - ```pip install -r requirements.txt```
 - Adjust **RMS_THRESHOLD_RANGE** and **RMS_THRESHOLD** - use **get_rms_values.py** for that, records few audio files,the goal is to cut off as much noise as possible but keep voice in audio, so play around with **LOCAL_RMS_THRESHOLD_RANGE** (within **get_rms_values.py**) and after that record some silence audios and play around with rms_threshold value ```print(rms_calc.is_voice_present(filtered_audio, rms_threshold=25))```

## 4.Project Structure
```
├── audio_predictor.py      # Main prediction module for audio commands
├── audio_tools.py          # Audio processing tools for calculating RMS and spectrograms
├── config.py               # Configuration settings
├── get_rms_values.py       # Script to calculate RMS values from audio files
├── logger.py               # Custom logger setup
├── simple_game.py          # Game script utilizing turtle graphics, controlled by voice commands
└── README.md               # Project documentation
```

## 5. Scripts
 - **config.py** - do I need to explain?
 - **audio_tools.py** - tools for preprocessing audio, calculating RMS and etc.
 - **get_rms_values.py** - used for testing
 - **audio_predictor.py** - it provides interface for recording audio, cleaning it and making prediction
 - **simple_game.py** - my turbo lil game :P
 - **logger.py** - logger :)

## 6. Training your own model
Use **SpeechToCommand.ipynb** Colab notebook, in the first cell adjust Tensorflow version to your needs (yeah, it's important :P), you can set execution environment's type to GPU (it will make training much faster) and then run whole notebook. I am not gonna dive into the details of neural network because nobody is gonna read ir use this anyways, but it's cnn structure becuase we are working with spectograms (so we do have some image-like data) - my notebook is based on https://www.tensorflow.org/tutorials/audio/simple_audio but I did it kinda "my way" and made neutal network a little bit more complicated.
Here are some results for model8:
![confusion_matrix](https://github.com/user-attachments/assets/41877bbe-bc61-46a5-aa62-2cc3dcf28dfb)
![train_hist](https://github.com/user-attachments/assets/594ceeb5-a711-48d7-8722-ce9782697a9e)




