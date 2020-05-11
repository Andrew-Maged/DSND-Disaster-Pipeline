# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disaster_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- notebooks
|- ETL Pipeline Preparation.ipynb # Building ETL Pipeline 
|- ML Pipeline Preparation.ipynb # Building ML Pipeline

- README.md
```

### Required packages:

- flask
- joblib
- jupyter # If you want to view the notebooks
- pandas
- plot.ly
- numpy
- scikit-learn
- sqlalchemy


### Data Overview:

The data in this project comes from a modified version of the figure-eight [disaster response data](https://www.figure-eight.com/dataset/combined-disaster-response-data/). In general it was pretty clean, the primary transform steps of the ETL are to a remove a few values that don't match up between the categories/messages data and remove a few bad remaining values.

This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.
