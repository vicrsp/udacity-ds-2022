from typing import List
import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
from tqdm import tqdm


BASE_FOLDER = 'data'
RAW_FOLDER = join(BASE_FOLDER, 'raw')
INTERIM_FOLDER = join(BASE_FOLDER, 'interim')

def list_folder_files(path: str) ->  List[str]:
    """Lists all files in a folder

    Args:
        path (str): The folder path

    Returns:
        List[str]: The list of files in the folder
    """
    return [f for f in listdir(path) if isfile(join(path, f))]

def read_file(path: str, file: str) -> pd.DataFrame:
    """Reads and parse a sensor reading file

    Args:
        path (str): The file path
        file (str): The file name

    Returns:
        pd.DataFrame: The file as a DataFrame
    """
    timestamp = datetime.strptime(file, '%Y.%m.%d.%H.%M.%S')
    df = pd.read_table(join(path, file), header=None, names=['Bearing 1','Bearing 2','Bearing 3','Bearing 4'], sep='\t')
    
    df['timestamp'] = timestamp
    df['timestamp_resampled'] = pd.date_range(timestamp, timestamp+timedelta(minutes=10), periods=df.shape[0]).tolist()

    return df

def folder_to_dataframe(folder_name: str) -> pd.DataFrame:
    """Reads all files in a folder and returns a single DataFrame

    Args:
        folder_name (str): The folder name

    Returns:
        pd.DataFrame: The DataFrame will all folder data
    """
    files = list_folder_files(folder_name)
    df = pd.concat([read_file(folder_name, f) for f in tqdm(files)])
    return df

if __name__ == '__main__':
    data_folders = ['1st_test', '2nd_test', '3rd_test']
    [folder_to_dataframe(join(RAW_FOLDER, folder)).to_parquet(join(INTERIM_FOLDER, f'{folder}.parquet'), allow_truncated_timestamps=True) for folder in tqdm(data_folders)]


