# Piano Composer

The Classical Piano Composer project is an AI-driven application designed to generate classical piano compositions in MIDI format. It uses a deep learning model trained on MIDI files, focusing on creating music with a single instrument.

## Features

- Train a neural network to learn patterns from classical piano music.
- Generate unique MIDI compositions using a trained model.
- Flexible and adaptable to various datasets of MIDI files containing a single instrument.

## Requirements

To run this project, ensure the following are installed:

- Python 3.x

### Required Python Packages:

- Music21
- Keras
- TensorFlow
- h5py

Install the dependencies using pip:

```bash
pip install music21 keras tensorflow h5py
```

## Training the Model

To train the neural network, use the `lstm.py` script. This script processes all MIDI files located in the `./midi_songs` directory to train the model.

> **Important:** Ensure that all MIDI files in the dataset use a single instrument for optimal training results.

### How to Train

Run the following command:

```bash
python lstm.py
```

### Key Notes:

- Training can be interrupted at any time. The model's weights from the most recently completed epoch will be saved automatically and can be used for generating music.
- Weights are stored in the `weights.hdf5` file, which serves as a checkpoint for resuming training or generating compositions.

## Generating Music

Once the model is trained, you can use the `predict.py` script to generate music compositions in MIDI format. The generated music will reflect patterns learned from the training dataset.

### How to Generate Music

Run the following command:

```bash
python predict.py
```

This script uses the saved weights (`weights.hdf5`) from the training process to generate MIDI compositions. The output file will be saved as `test_output.mid` in the project directory.

## Directory Structure

- `./midi_songs/` - Contains the training dataset of MIDI files.
- `weights.hdf5` - Stores the trained model weights.
- `lstm.py` - Script to train the neural network.
- `predict.py` - Script to generate music using the trained model.

## Getting Started

1. Prepare a dataset of MIDI files and save them in the `./midi_songs` directory.
2. Train the neural network using `lstm.py`.
3. Generate compositions using `predict.py`.

## Acknowledgments

This project leverages the capabilities of LSTM (Long Short-Term Memory) networks, a type of recurrent neural network, for music sequence modeling and generation. The implementation is powered by Keras and TensorFlow.
