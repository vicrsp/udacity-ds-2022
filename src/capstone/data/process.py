from typing import List, Tuple
import pandas as pd
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
from data.utils import reduce_mem_usage
from tqdm import tqdm

class TEPDataLoader:
    def __init__(self, path) -> None:
        self.path = path

    def _load(self, name) -> pd.DataFrame:
        chunksize = 10 ** 5
        df = pd.DataFrame()
        with pd.read_csv(join(self.path, name), sep=',', chunksize=chunksize) as reader:
            for chunk in tqdm(reader):
                df = pd.concat([df, reduce_mem_usage(chunk)])
        return reduce_mem_usage(df)

    def load_training_datasets(self) -> Tuple[pd.DataFrame]:
        df_faulty = self._load('faulty_training.csv')
        df_normal = self._load('fault_free_training.csv')
        return df_faulty, df_normal

    def load_testing_datasets(self) -> Tuple[pd.DataFrame]:
        df_faulty = self._load('faulty_testing.csv')
        df_normal = self._load('fault_free_testing.csv')
        return df_faulty, df_normal