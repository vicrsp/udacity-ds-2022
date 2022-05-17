import sys
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.autoencoder import LSTMAutoencoder, LSTMAutoencoderTransformer
from data.process import TEPDataLoader
import joblib
from os.path import join

def load_data(path: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        path (str): The file folder path

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: The input and output variables, and the list of categories
    """
    loader = TEPDataLoader(path)
    return loader._load(name)

def process_data(X: pd.DataFrame, window, simulation_size):
    pv_columns = X.drop(columns=['xmeas_9', 'xmeas_18', 'xmeas_7', 'xmeas_13', 'xmv_11', 'xmv_9', 'xmv_7', 'xmv_6', 'xmv_5','xmv_3']).columns[3:]
    # get the training and testing datasets
    X_train, X_val = LSTMAutoencoderTransformer.train_test_split(X)
    # create the transformer
    transformer = LSTMAutoencoderTransformer(selected_columns=pv_columns)
    # fit the transformer
    X_train_transf = transformer.fit_transform(X_train)
    X_val_transf = transformer.transform(X_val)
    # create the training dataset
    X_train_windowed = LSTMAutoencoder.created_windowed_dataset(X_train_transf, window=window, batch_size=simulation_size)
    X_val_windowed = LSTMAutoencoder.created_windowed_dataset(X_val_transf, window=window, batch_size=simulation_size)
    
    return transformer, X_train_windowed, X_val_windowed


def save_model(model: LSTMAutoencoder, transformer: LSTMAutoencoderTransformer, model_filepath: str) -> None:
    """Saves the model to a file

    Args:
        model (Pipeline): The trained model
        model_filepath (str): The file to save the model
    """
    # saving transformer
    joblib.dump(transformer, join(model_filepath, 'transformer.pkl'))
    # saving model
    model.save(model_filepath, 'model')

def inverse_transform(X, batch_size=500, window=1, transformer=None):
    if(transformer is not None):
        return transformer.inverse_transform(LSTMAutoencoder.reverse_windowed_dataset(X, batch_size, window))
    return LSTMAutoencoder.reverse_windowed_dataset(X, batch_size, window)

def predict_anomalies(X, model, transformer, threshold, batch_size, window=1):
    simulation_failures = []
    for (fault_number, simulation_run), df_sim in tqdm(X.groupby(['faultNumber','simulationRun'])):
        X_transf = transformer.transform(df_sim)
        # create the training dataset
        X_windowed = LSTMAutoencoder.created_windowed_dataset(X_transf, window=window, batch_size=batch_size)
        # predict each signal
        X_pred = pd.DataFrame(inverse_transform(model.predict(X_windowed), batch_size=batch_size, window=window), columns=transformer.selected_columns)
        X_transf = pd.DataFrame(X_transf, columns=transformer.selected_columns) 
        loss_mae = np.mean(np.abs(X_pred - X_transf), axis=1)

        failure_index = df_sim.reset_index().loc[loss_mae>threshold]['sample']
        if(len(failure_index)>0):
            idx = failure_index.values[0]
            simulation_failures.append((fault_number, simulation_run, idx, idx*3, True))
        else:
            simulation_failures.append((fault_number, simulation_run, np.nan, np.nan, False))
        
    simulation_failures = pd.DataFrame(simulation_failures, columns=['failureType','simulationRun', 'failure_index', 'failure_time','anomaly'])
    return simulation_failures


def load_model(path: str):
    # load the trained model
    model = LSTMAutoencoder(f'{path}/model_keras.h5')
    # import the transformer
    transformer = joblib.load(f'{path}/transformer.pkl')

    return model, transformer

def save_results(filename, model_filepath, results):
    results.to_csv(f'{model_filepath}/predictions_{filename}', index=False)

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
        results = predict_anomalies(X, model, transformer, threshold=threshold, batch_size=batch_size, window=1)

        print('Saving results...\n  path: {}'.format(model_filepath))
        save_results(filename, model_filepath, results)        

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'evaluate_anomaly_detection.py data/raw faulty_training.csv models/final')



if __name__ == '__main__':
    main() 