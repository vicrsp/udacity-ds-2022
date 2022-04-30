import numpy as np
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras import regularizers
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from os.path import join

class LSTMAutoencoder(BaseEstimator, RegressorMixin):
    def __init__(self, save_path=None, epochs=100, batch_size=32, validation_split=0.1, dropout=0.2) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dropout = dropout
        self.save_path = save_path
        
    def _build_network(self, X: np.ndarray) -> Model:
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, dropout=self.dropout)(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False, dropout=self.dropout)(L1)
        # The RepeatVector layer simply repeats the input n times.
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True, dropout=self.dropout)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True, dropout=self.dropout)(L4)
        # The TimeDistributed layer creates a vector with a length of the number of outputs from the previous layer. 
        output = TimeDistributed(Dense(X.shape[2]))(L5)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mae')
        return model

    def fit(self, X: np.ndarray, y=None):   
        if(self.save_path is None):
            self._model = self._build_network(X)
            self._history = self._model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
                                            validation_split=self.validation_split)
        
        else:
            self._load()
        return self

    def set_threshold(self, error_distribution, percentile=99) -> None:
        self._anomaly_threshold = np.percentile(error_distribution, percentile)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def reconstruction_error(X_in, X_out) -> np.array:
        return np.mean(np.abs(X_in - X_out))

    def predict_anomaly(self, X) -> np.ndarray:
        X_out = LSTMAutoencoderTransformer.reshape_predictions(self.predict(X))
        error = self.reconstruction_error(X, X_out)
        return error > self._anomaly_threshold

    def save(self, path, name) -> None:
        self._model.save(join(path, f'{name}.h5'))

    def _load(self) -> None:
        self._model = load_model(self.save_path)

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