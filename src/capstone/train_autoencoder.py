import sys
from typing import Tuple
import numpy as np
import pandas as pd
from model.autoencoder import LSTMAutoencoder, LSTMAutoencoderTransformer
from data.process import TEPDataLoader
import joblib
from os.path import join

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        path (str): The file folder path

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: The input and output variables, and the list of categories
    """
    loader = TEPDataLoader(path)
    return loader._load('fault_free_training.csv')

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


def build_model(**params) -> LSTMAutoencoder:
    """Builds the pipeline used to train the model.

    Returns:
        GridSearchCV: A grid search cross-validation object using a pipeline
    """
    return LSTMAutoencoder(**params)

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


def main():
    if len(sys.argv) == 3:
        filepath, model_filepath = sys.argv[1:]
        print('Loading training data...\n    path: {}'.format(filepath))
        X_normal = load_data(filepath)
         
        print('Processing data...')
        transformer, X_train, X_val = process_data(X_normal, window=1, simulation_size=500)

        print('Building model...')
        model = build_model(epochs=30, batch_size=64, dropout=0.3, validation_data=(X_val, X_val))
        
        print('Training model...')
        model.fit(X_train, None)
        
        print('Saving model...\n  path: {}'.format(model_filepath))
        save_model(model, transformer, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_autoencoder.py ../data/raw ..data/models')


if __name__ == '__main__':
    main() 