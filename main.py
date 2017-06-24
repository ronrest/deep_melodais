"""
DEEP MELODAIS
- An AI based on deep neural networks that that composes music.
- uses an LSTM for generating monophonic music as MIDI instructions.

Basing this on the work described in the following places:
- http://yoavz.com/music_rnn/
- https://www.youtube.com/watch?v=S_f2qV2_U00


MUSIC TERMINOLOGY:
- Melody: The main notes playing the tune.
- Harmony: The support notes?
- Monophonic: only one note is played at any particular timestep.

MIDI BACKGROUND:
- 88 possible notes from A0 to C8 in a MIDI file.
  So that will be the vocabulary for the melody.


ASSUMPTIONS OF THE INPUT:
- The melody is monophonic.
- The harmony at each time step can be mapped to a chord class.
  This means, we there will be multiple notes playing at each time step,
  but the combinations are common enough to have labels.
  eg:
    A "C Major" chord involves the notes C, E, G playing at once.
  This will allow us to represent the harmony as a single item per timestep.


ARCHITECTURE:
- http://yoavz.com/music_rnn/img/dual_softmax_fig.png
- We will have two softmax functions for the output
- This allows us to take the sum of two softmax functions as the loss
  function for our model.


"""

DATA_DIR = "data"

