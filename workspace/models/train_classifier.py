# import libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import sqlite3
import re
import sys
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import xgboost as xgb   #may have to install first with updated pip
from xgboost import XGBClassifier


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split  
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("maxent_ne_chunker")
nltk.download("wordnet")

def removeStopWords(sentence):
    #remove stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two','three','four','five',
                       'six','seven','eight','nine','ten','may',
                       'also','across','among','beside','however',
                       'yet','within'])
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    return re_stop_words.sub(" ", sentence)

def cleanHtml(sentence):
    #function to clean HTML
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): 
    #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def load_data(database_filepath):
    #loading data, the path should be indicated when calling "main" function
    # load to database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Message", engine)

    X = df.message
    # clean data by calling functions
    X = X.apply(removeStopWords)
    X = X.apply(cleanHtml)
    X = X.apply(cleanPunc)  
    X = X.apply(keepAlpha)
    
    catetories_col_names = [col for col in list(df.columns) if col not in ['index','id','message','original','genre']]
    Y = df[catetories_col_names]
    
    return X, Y, catetories_col_names

def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    

def build_model():
    # text processing and model pipeline

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(xgb.XGBClassifier()))
    ])
    # define parameters for GridSearchCV
    parameters= {
        'clf__estimator__eta': [0.1,0.3,0.4],
        'clf__estimator__max_depth': [2,4],
    }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters)    
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred,target_names = category_names))
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))

def save_model(model, model_filepath):
    """
    Saves model as a .pkl file. Destination is set by model_filepath argument.
    
    Arguments:
    model: trained estimator to save
    model_filepath: destination for model save
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        # fit model
        print('Training model...')        
        model.fit(X_train, Y_train)
        # output model test results
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