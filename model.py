#define Model
class Model():
    def __init__(self):
        # --- Define model ---
         Sequential([
         LSTM(100, activation='tanh', input_shape=(24, 1)),  # encoder
         RepeatVector(12),  # repeat context 12 times
         LSTM(100, activation='tanh', return_sequences=True),  # decoder
         TimeDistributed(Dense(1))  # 12 outputs
         ])

         model.compile(loss='mse', optimizer='adam')
