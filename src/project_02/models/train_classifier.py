import sys
import re
from typing import List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.engine import create_engine
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """_summary_

    Args:
        database_filepath (str): The database file path

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: The input and output variables, and the list of categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('message_categories', engine)
    
    target_columns = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']

    X = df.message.astype(str).values
    Y = df[target_columns].astype(int).values
    return X, Y, target_columns

def tokenize(text: str) -> List[str]:
    """Tokenizes a given text, applying the following steps:
        1. Remove stop words
        2. Lemmatize
        3. Convert to lower case and strip whit spaces  
    Args:
        text (str): The text to tokenize

    Returns:
        List[str]: The list of tokens extracted
    """
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words    
    stop_words = stopwords.words("english")
    filtered_words = [word for word in tokens if word not in stop_words]
    
    # Lemmatize and clean text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in filtered_words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model() -> GridSearchCV:
    """Builds the pipeline used to train the model.

    Returns:
        GridSearchCV: A grid search cross-validation object using a pipeline
    """
    # instantiate transformers and classifiers
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier(random_state=0)
    vt = VarianceThreshold()
    
    # create the pipeline
    pipeline = Pipeline([
                ('text_pipeline', Pipeline([
                    ('vect', vect),
                    ('tfidf', tfidf)
                ])),
                ('variance_threshold', vt),
                ('clf', MultiOutputClassifier(clf, n_jobs=-1))
                ])
    
 
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__max_depth': [2, 5, None],
        'text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
    }
    
    cv = GridSearchCV(pipeline, parameters, verbose=3, random_state=0)
    return cv

def evaluate_model(model: Pipeline, X_test: np.ndarray, Y_test: np.ndarray, category_names: List[str]) -> None:
    """Displays the model metrics for a given dataset

    Args:
        model (Pipeline): The trained model
        X_test (np.ndarray): The input dataset
        Y_test (np.ndarray): The target values
        category_names (List[str]): The category names
    """
    y_pred = model.predict(X_test)
    for i, value in enumerate(category_names):    
        print(f'Category: {value}')
        print(classification_report(Y_test[i], y_pred[i]))


def save_model(model: Pipeline, model_filepath: str) -> None:
    """Saves the model to a file

    Args:
        model (Pipeline): The trained model
        model_filepath (str): The file to save the model
    """
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model.best_estimator_, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()