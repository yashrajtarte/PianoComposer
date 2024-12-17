import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Main function to train the music generation neural network
def train_network():
    """ Train a Neural Network to generate music """
    # Extract notes from MIDI files
    notes = get_notes()
    # Get the total number of unique pitch names
    n_vocab = len(set(notes))
    # Prepare the input and output sequences for the network
    network_input, network_output = prepare_sequences(notes, n_vocab)
    # Create the LSTM-based neural network
    model = create_network(network_input, n_vocab)
    # Train the network using the prepared sequences
    train(model, network_input, network_output)

# Function to extract notes and chords from MIDI files
def get_notes():
    """ Get all the notes and chords from the MIDI files in the ./midi_songs directory """
    notes = []
    for file in glob.glob("Classical-Piano-Composer/midi_songs/*.mid"):
        midi = converter.parse(file)  # Parse the MIDI file
        print("Parsing %s" % file)
        try:
            # If the MIDI file has instrument parts, extract the first part
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            # If the file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        
        # Extract notes and chords
        for element in notes_to_parse:
            if isinstance(element, note.Note):  # If it's a note, add its pitch
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):  # If it's a chord, add its normal order
                notes.append('.'.join(str(n) for n in element.normalOrder))
    
    # Save the extracted notes for later use
    with open('Classical-Piano-Composer/data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    return notes

# Function to prepare input and output sequences for training
def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100  # Length of each input sequence
    # Get a sorted list of unique pitch names
    pitchnames = sorted(set(item for item in notes))
    # Map each pitch name to an integer
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []  # Input sequences
    network_output = []  # Corresponding output notes

    # Create sequences of notes and corresponding next notes
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Number of patterns
    n_patterns = len(network_input)
    # Reshape input into the format required by LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input to range [0, 1]
    network_input = network_input / float(n_vocab)
    # One-hot encode the output
    network_output = to_categorical(network_output)
    return network_input, network_output

# Function to create the neural network
def create_network(network_input, n_vocab):
    """ Create the structure of the neural network """
    model = Sequential()
    # Add LSTM layers with dropout for regularization
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    # Add batch normalization for faster convergence
    model.add(BatchNorm())
    # Add dropout to prevent overfitting
    model.add(Dropout(0.3))
    # Dense layer with ReLU activation
    model.add(Dense(256))
    model.add(Activation('relu'))
    # Another batch normalization and dropout
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    # Output layer with softmax activation
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    # Compile the model with categorical crossentropy loss and RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # Print model summary
    model.summary()
    return model

# Function to train the neural network
def train(model, network_input, network_output):
    """ Train the neural network """
    # Define a checkpoint to save the model with the lowest loss
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # Train the model using the prepared sequences
    model.fit(network_input, network_output, epochs=20, batch_size=128, callbacks=callbacks_list)

# Entry point of the script
if __name__ == '__main__':
    train_network()
