import sys
import re
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
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

    X = df.message.values
    Y = df[target_columns].values
    
    return X, Y, target_columns

def tokenize(text):
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatize and clean text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    # Remove stop words    
    stop_words = stopwords.words("english")
    filtered_words = [word for word in clean_tokens if word not in stop_words]
    return filtered_words


def build_model():
    # instantiate transformers and classifiers
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier()
    
    # create the pipeline
    pipeline = Pipeline([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
    
    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_cv = model.best_estimator_.predict(X_test)
    for i, value in enumerate(category_names):    
        print(f'Category: {value}')
        print(classification_report(Y_test[i], y_pred_cv[i]))


def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)

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
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()