import json
import numpy as np

import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re

from sklearn.feature_extraction.text import CountVectorizer
import random

app = Flask(__name__)


def tokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def get_class_distribution_graph():
    """Draws a graph with the message category counts

    Returns:
        dict: The graph object
    """
    message_count =  df.iloc[:, 4:].astype(int).apply(sum).sort_values(ascending=False)
    message_types = message_count.index
    
    graph ={
        'data': [
                Bar(
                    y=message_count,
                    x=message_types
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    #'title': "Class",
                    'tickangle': 45,
                }
            }
    }

    return graph


def get_word_cloud_graph(n_words = 20):
    """Draws a word cloud graph

    Args:
        n_words (int, optional): The number of top words to display. Defaults to 20.

    Returns:
        dict: The graph object
    """
    # create the bag of words    
    vect = CountVectorizer(tokenizer=tokenize, max_features=n_words)
    X = vect.fit_transform(df.message)
    # extract the words and frquecies
    bag_of_words = pd.DataFrame(list(zip(vect.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())), columns=['word', 'freq'])
    # define the plot colors
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(n_words)]
    
    graph = {
        'data': [
                Scatter(
                    y=random.choices(range(n_words), k=n_words),
                    x=random.choices(range(n_words), k=n_words),
                    text=bag_of_words.word,
                    mode='text',
                    textfont={'size': bag_of_words.freq / 100, 'color': colors}
                )
            ],

            'layout': {
                'title': 'Most Frequent Tokens',
            }
    }

    return graph

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    graphs = [
        get_class_distribution_graph(),
        get_word_cloud_graph()
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()