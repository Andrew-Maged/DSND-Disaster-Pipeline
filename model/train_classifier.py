import sys

#!pip install scikit-learn --upgrade
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Loads data from the Disaster Response Database and returns X, Y and Classes names
    
    Args:
        database_filepath(str): database file path
        
    Returns:
        X (Pandas Dataframe): independent features
        Y (Pandas Dataframe): Classes
        category_names (List): Classes names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    # I found that there are messages with null responses, so i will drop those rows
    df.dropna(subset=['related'],inplace=True) 
    X = df[['message','genre']]
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y,category_names

    
def tokenize(text):
    '''
    Simple tokenizer that lemmatize after applying word tokenization
    Args:
        text (str): input messages
        
    Returns:
        tokens (list): list of lemmatized tokens
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def build_model():
        
    Text_pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer())])


    Column_Transformer = ColumnTransformer(
                transformers=[('Categorical_Transform', OneHotEncoder(), ['genre']),
                              ('Text_Transform',Text_pipeline,'message')
                             ])

    pipeline = Pipeline([
            ('ColumnTransformer', Column_Transformer ),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
   
    #Grid Search model optimization
    parameters = {'clf__estimator__bootstrap': (True,False),
                  'ColumnTransformer__Text_Transform__tfidf__smooth_idf': (True,False),
                  'clf__estimator__n_estimators': [50,100,200]
                  #,'clf__estimator__criterion' :['gini','entropy']
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters,cv=2, verbose=1, n_jobs=-1)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints out Multi-Output Classifier results
    Args:
        model (Scikit-learn Model): Scikit-learn fitted model
        X_test (Pandas Dataframe): Independent Features for training
        Y_test (Pandas Dataframe): Actual Classes
        category_names (List): Classes Names
    Returns:
        None
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """Save a snapshot of the trained model as pickle
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): Model save location
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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