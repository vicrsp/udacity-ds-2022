import sys
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.autoencoder import LSTMAutoencoder, LSTMAutoencoderTransformer
from data.process import TEPDataLoader
import joblib
from os.path import join


def load_data(path: str, name: str) -> pd.DataFrame:
    """Loads the data

    Args:
        path (str): The file folder path
        name (str): The file name

    Returns:
        pd.DataFrame: The loaded data
    """
    loader = TEPDataLoader(path)
    return loader._load(name)


def inverse_transform(X: np.ndarray, simulation_size=500, window=1, transformer: LSTMAutoencoderTransformer = None):
    """Custom inverse transform function

    Args:
        X (np.ndarray): The data to transform
        simulation_size (int, optional): The simulation size. Defaults to 500.
        window (int, optional): The sliding window size. Defaults to 1.
        transformer (LSTMAutoencoderTransformer, optional): Tje transformer. Defaults to None.

    Returns:
        _type_: _description_
    """

    if(transformer is not None):
        return transformer.inverse_transform(LSTMAutoencoder.reverse_windowed_dataset(X, simulation_size, window))
    return LSTMAutoencoder.reverse_windowed_dataset(X, simulation_size, window)


def predict_anomalies(X: pd.DataFrame, model: LSTMAutoencoder, transformer: LSTMAutoencoderTransformer, threshold: int, simulation_size: int, window=1) -> pd.DataFrame:
    """Predict anomalies in the dataset

    Args:
        X (pd.DataFrame): Input data
        model (LSTMAutoencoder): The trained model
        transformer (LSTMAutoencoderTransformer): The transformer
        threshold (int): The anomaly threshold
        batch_size (int): The simulation size
        window (int, optional): The sliding window size. Defaults to 1.

    Returns:
        pd.DataFrame: The anomaly prediction
    """
    simulation_failures = []
    for (fault_number, simulation_run), df_sim in tqdm(X.groupby(['faultNumber', 'simulationRun'])):
        X_transf = transformer.transform(df_sim)
        # create the training dataset
        X_windowed = LSTMAutoencoder.created_windowed_dataset(
            X_transf, window=window, batch_size=simulation_size)
        # predict each signal
        X_pred = pd.DataFrame(inverse_transform(model.predict(
            X_windowed), simulation_size=simulation_size, window=window), columns=transformer.selected_columns)
        X_transf = pd.DataFrame(X_transf, columns=transformer.selected_columns)
        loss_mae = np.mean(np.abs(X_pred - X_transf), axis=1)

        failure_index = df_sim.reset_index(
        ).loc[loss_mae > threshold]['sample']
        if(len(failure_index) > 0):
            idx = failure_index.values[0]
            simulation_failures.append(
                (fault_number, simulation_run, idx, idx*3, True))
        else:
            simulation_failures.append(
                (fault_number, simulation_run, np.nan, np.nan, False))

    simulation_failures = pd.DataFrame(simulation_failures, columns=[
                                       'failureType', 'simulationRun', 'failure_index', 'failure_time', 'anomaly'])
    return simulation_failures


def load_model(path: str):
    """Loads the model and transformer from a file

    Args:
        path (str): The file path

    Returns:
        Tuple: The loaded model and transformer
    """
    # load the trained model
    model = LSTMAutoencoder(f'{path}/model_keras.h5')
    # import the transformer
    transformer = joblib.load(f'{path}/transformer.pkl')

    return model, transformer


def save_results(filename: pd.DataFrame, save_path: str, results: str):
    """Save the results to a file

    Args:
        filename (str): The file name
        model_filepath (str): The save path
        results (pd.DataFrame): The results to save
    """
    results.to_csv(f'{save_path}/predictions_{filename}', index=False)


def main():
    if len(sys.argv) == 4:
        filepath, filename, model_filepath = sys.argv[1:]
        print('Loading model...\n  path: {}'.format(model_filepath))
        model, transformer = load_model(model_filepath)

        print('Loading data...\n    path: {}'.format(filepath))
        X = load_data(filepath, filename)

        print('Predicting anomalies...')
        threshold = 1.2609543704986577
        batch_size = X['sample'].unique().shape[0]
        results = predict_anomalies(
            X, model, transformer, threshold=threshold, simulation_size=batch_size, window=1)

        print('Saving results...\n  path: {}'.format(model_filepath))
        save_results(filename, model_filepath, results)

    else:
        print('Invalid arguments, please check. '
              '\n\nExample: python '
              'evaluate_anomaly_detection.py data/raw faulty_training.csv models/final')


if __name__ == '__main__':
    main()
