import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load and combine the messages and categories datasets into one dataframe
    
    Args:
        messages_filepath (str): The path to the messages file
        categories_filepath (str): The path to the categories file
    
    Returns:
        df (Pandas Dataframe): The combined messages and categories dataframe
        
    '''
    
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath) 
    
    return messages.merge(categories, on='id')


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    
    categories.columns = category_colnames

    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df,categories],axis=1 ) 

    #Remove duplicates
    df.drop_duplicates(subset='message', inplace=True)
    df.dropna(subset=category_colnames, inplace=True)
    return df
    
def save_data(df, database_filename):
    '''
    Save the new Dataframe to a sqlite database
    
    Args:
        df (Pandas Dataframe): The cleaned dataframe
        database_filename (string): The path to the new database file
        
    Returns:
        None
    '''  
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    engine.dispose()
    
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