import pickle
import numpy
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical

def generate():
    """ Generate a piano MIDI file """
    # Load the notes used to train the model
    with open('Classical-Piano-Composer/data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all unique pitch names
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))  # Total number of unique notes and chords

    # Prepare input sequences and normalize them
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    # Create the LSTM-based model
    model = create_network(normalized_input, n_vocab)
    # Generate notes using the trained model
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    # Create a MIDI file from the generated notes
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # Map between notes and integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100  # Define the length of input sequences
    network_input = []  # Stores the input sequences
    output = []  # Stores the corresponding output notes

    # Generate input-output pairs for training
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input to range [0, 1]
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

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
    # Add Batch Normalization for better convergence
    model.add(BatchNormalization())
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.3))
    # Add a Dense layer with ReLU activation
    model.add(Dense(256))
    model.add(Activation('relu'))
    # Add another Batch Normalization and Dropout layer
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # Add the output layer with Softmax activation
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    # Compile the model with categorical crossentropy loss and RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # Print model summary
    model.summary()
    try:
        # Load pre-trained weights if available
        model.load_weights('Classical-Piano-Composer/weights.hdf5', by_name=True)
    except ValueError as e:
        print(f"Error loading weights: {e}")

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random sequence from the input as the starting point
    start = numpy.random.randint(0, len(network_input) - 1)

    # Map integers back to notes
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # Extract the chosen sequence
    pattern = network_input[start]
    prediction_output = []  # Store the generated notes

    # Generate 500 notes
    for note_index in range(500):
        # Reshape input to match the model's expected input shape
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        # Normalize the input
        prediction_input = prediction_input / float(n_vocab)

        # Predict the next note
        prediction = model.predict(prediction_input, verbose=0)
        index = numpy.argmax(prediction)  # Get the note with the highest probability
        result = int_to_note[index]  # Convert the index back to a note
        prediction_output.append(result)  # Add the note to the output

        # Update the input pattern for the next prediction
        pattern.append(index)
        pattern = pattern[1:len(pattern)]  # Remove the first element to maintain sequence length

    return prediction_output

def create_midi(prediction_output):
    """ Convert the predicted notes into a MIDI file """
    offset = 0  # Start at the beginning
    output_notes = []  # List to hold the generated notes and chords

    # Create note and chord objects based on the generated patterns
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():  # Pattern is a chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:  # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increment offset to prevent stacking of notes
        offset += 0.5

    # Create a MIDI stream from the notes and chords
    midi_stream = stream.Stream(output_notes)

    # Write the stream to a MIDI file
    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
