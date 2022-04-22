import numpy as np
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

class LSTMAutoencoder(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=100, batch_size=32, validation_split=0.1) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        pass

    def _build_network(self, X: np.ndarray) -> Model:
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True)(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)

        output = TimeDistributed(Dense(X.shape[2]))(L5)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mae')
        return model

    def fit(self, X: np.ndarray, y=None):        
        self._model = self._build_network(X)
        self._history = self._model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
                                        validation_split=self.validation_split)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    # @staticmethod
    # def reshape_input(X, time_steps=1):
    #     # reshape inputs for LSTM [samples, timesteps, features]
    #     return X.reshape(X.shape[0], time_steps, X.shape[1])

    # @staticmethod
    # def reshape_predictions(X):
    #     return X.reshape(X.shape[0], X.shape[2])


class LSTMAutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, time_steps=1) -> None:
        self.time_steps = time_steps
    
    def fit(self, X):
        return self

    @staticmethod
    def reshape_input(X, time_steps=1):
        # reshape inputs for LSTM [samples, timesteps, features]
        return X.reshape(X.shape[0], time_steps, X.shape[1])

    @staticmethod
    def reshape_predictions(X):
        return X.reshape(X.shape[0], X.shape[2])
    
    def transform(self, X):
        return LSTMAutoencoderTransformer.reshape_input(X, self.time_steps)
    
    def inverse_transform(self, X):
        return LSTMAutoencoderTransformer.reshape_predictions(X)