from unicodedata import name
import numpy as np
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras import regularizers
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from os.path import join
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
class LSTMAutoencoder(BaseEstimator, RegressorMixin):
    def __init__(self, save_path=None, epochs=100, batch_size=32, validation_data=None, validation_split=0.1,
                    dropout=0.3, lstm_units=64) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.validation_split = validation_split
        self.dropout = dropout
        self.save_path = save_path
        self.lstm_units = lstm_units

        if(self.save_path is not None):
             self._load()

    def _build_network(self, X: np.ndarray) -> Model:
        # Creates the encoder layer
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        encoder = LSTM(self.lstm_units, activation='relu', dropout=self.dropout, name='encoder')(inputs)
        
        # Creates the decoder layers
        # The RepeatVector layer simply repeats the input n times.
        decoder = RepeatVector(X.shape[1])(encoder)
        decoder = LSTM(self.lstm_units, activation='relu', return_sequences=True, dropout=self.dropout, name='decoder')(decoder)
        # The TimeDistributed layer creates a vector with a length of the number of outputs from the previous layer. 
        output = TimeDistributed(Dense(X.shape[2]))(decoder)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X: np.ndarray, y=None):   
        self._model = self._build_network(X)
        if(self.validation_data is not None):
            self._history = self._model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
                                        validation_data=self.validation_data, shuffle=True)
        else:
            self._history = self._model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size,
                                        validation_split=self.validation_split, shuffle=True)
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
        self._model.save(join(path, f'{name}_keras.h5'))
        joblib.dump(self._history, join(path, f'{name}_history.pkl'))

    def _load(self) -> None:
        self._model = load_model(self.save_path)
        self._history = joblib.load(self.save_path.replace('_keras.h5', '_history.pkl'))

    def plot_model(self) -> None:
        plot_model(self._model)
    
    def plot_history(self) -> None:
        if(self._history is None): return

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(self._history.history['loss'])
        ax.plot(self._history.history['val_loss'])
        ax.set_title('Training History')
        ax.set_ylabel('MSE')
        ax.set_xlabel('epoch')
        ax.legend(['Train', 'Validation'])
        fig.show()

    @staticmethod
    def created_windowed_dataset(X, batch_size=500, window=1):
        n = X.shape[0]
        X_s = []
        for b in range(0, n, batch_size):
            for i in range(batch_size - window + 1):
                v = X[(b+i):(b + i + window), :]
                X_s.append(v)
            
        return np.array(X_s)

    @staticmethod
    def reverse_windowed_dataset(X, batch_size=500, window=1):        
        n = X.shape[0]
        X_s = []
        for i in range(0, n, batch_size-window+1):
            for k in range(i, i+batch_size-window):
                X_s.append(X[k, 0, :])
            for j in range(window):
                X_s.append(X[k+1, j, :])

        return np.array(X_s)

class LSTMAutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns) -> None:
        self.selected_columns = selected_columns
  
    @staticmethod
    def train_test_split(X, split_size=0.2):
            simulations = X.simulationRun.unique()
            simulations_shuffled = np.random.permutation(simulations)
            train_simulations = simulations_shuffled[:int(len(simulations)*(1-split_size))]
            test_simulations = simulations_shuffled[int(len(simulations)*(1-split_size)):]

            return X.loc[X.simulationRun.isin(train_simulations), :], X.loc[X.simulationRun.isin(test_simulations)]

    def fit(self, X):
        self._scaler = StandardScaler().fit(X[self.selected_columns])
        return self

    def transform(self, X):
        # scale the variables
        X_scaled = self._scaler.transform(X[self.selected_columns])
        return X_scaled
    
    def inverse_transform(self, X):
        return self._scaler.inverse_transform(X[self.selected_columns])