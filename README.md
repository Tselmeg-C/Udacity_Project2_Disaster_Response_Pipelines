# Udacity_Project2_Disaster_Response_Pipelines
key words: NLP, tfidf, Pipeline, GridSearch, Multi-lable classification, Flask, Plotly

## Project description
In this project, data engineering skills were applied to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages.

In the project repo, messages.csv contains real messages that were sent during disaster events, while the categories.csv contains category information of those messages. One piece of message can be labled in more categories. A machine learning pipeline should be created to categorize these events so that they could be sent to an appropriate disaster relief agency, which is the real world implication of this project.

A web app were created, where a new message can be classified in several categories. The web app also displaies three visualizations of the data. 

## Project components
There are three components in this project.

1. ETL Pipeline
In the preparation phase, data were processed in Jupyter Notebook, refer 'ETL Pipeline Preparation.ipynb' for details. A data cleaning pipeline 'process_data.py' (in Workspace/Data folder) was created in a Python script including following functions:

⋅⋅*Loads the messages and categories datasets
⋅⋅*Merges the two datasets
⋅⋅*Cleans the data
⋅⋅*Stores it in a SQLite database

2. Machine Learning Pipeline
In the preparation phase, data were processed in Jupyter Notebook, refer 'ML Pipeline Preparation.ipynb' for details. A machine learning  pipeline 'train_classifier.py' (in Workspace/Model folder) was created in a Python script including following functions:

⋅⋅*Loads data from the SQLite database
⋅⋅*Splits the dataset into training and test sets
⋅⋅*Builds a text processing and machine learning pipeline
⋅⋅*Trains and tunes a model using GridSearchCV
⋅⋅*Outputs results on the test set
⋅⋅*Exports the final model as a pickle file

3. Flask Web App
Here's the file structure of the project:

'''
- app
| - template
| |- master.html      # main page of web app
| |- go.html          # classification result page of web app
|- run.py             # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv    # data to process
|- process_data.py          # data processing module
|- InsertDatabaseName.db    # database to save clean data to

- models
|- train_classifier.py  # model traning module
|- classifier.pkl       # saved model 

- README.md
'''
