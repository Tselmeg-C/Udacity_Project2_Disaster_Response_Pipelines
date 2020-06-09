import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    #plot_1: genre counts       
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #plot_2: number of messages in each category
    categories = list(df.iloc[:,4:].columns.values)
    frequency = df.iloc[:,4:].sum().values
    
    #plot_3: number of messages with multiple lables
    rowSums = df.iloc[:,2:].sum(axis=1)
    multiLabel_counts = rowSums.value_counts()
    multiLabel_counts = multiLabel_counts.iloc[1:]
    
    # create 3 plots and add in graphs list
    graphs = [
        # plot_1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"                
                }
                
            }
        },
        # plot_2
        {
            'data': [
                Bar(
                    x=categories,
                    y=frequency                
                )
            ],
            
            'layout': {
                 'title': 'Messages in Each Category',
                 'xaxis': {
                      'title': "Message Type", 
                  },
                 'yaxis': {
                      'title': "Number of messages"
                  }
                  
            }
        
        },
        # plot_3
        {
            'data': [
                Bar(
                    x=multiLabel_counts.index,
                    y=multiLabel_counts.values                
                )
            ],
            
            'layout': {
                'title': 'Messages having multiple labels',
                'xaxis': {
                    'title': "Number of labels" 
                  },
                'yaxis': {
                    'title': "Number of messages"
                  }
            } 
        }    
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()