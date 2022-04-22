import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Loads the data from .csv files

    Args:
        messages_filepath (str): The messages file path
        categories_filepath (str): The categories file path

    Returns:
        pd.DataFrame: A dataset containing the raw messages and categories data
    """
    # load the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the datasets
    df = messages.merge(categories, left_on='id', right_on='id')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Cleans the raw dataset

    Args:
        df (pd.DataFrame): The raw data

    Returns:
        pd.DataFrame: The cleaned data
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    category_colnames = [x.split('-')[0] for x in row]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

    # Drop the  categories  from `df`
    df.drop('categories', axis = 1, inplace = True)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # remove "related" columns rows with a value of 2
    df = df[df['related'] != 2]

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """Saves the processed data

    Args:
        df (pd.DataFrame): The processed data
        database_filename (str): The database file name
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('message_categories', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()