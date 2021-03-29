import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the data sources.
    
    Input
    -----
    messages_filepath: string
        path of messages dataset csv file
    categories_filepath: string
        path of messages categories dataset csv file
    
    Output
    ------
    df: Pandas DataFrame
        messages and categories dataframe
    """
    
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on="id", how='inner')
    
    return df


def clean_data(df):
    """
    Cleans the input dataframe.
    Turns the categories column into separate columns for each category having values of just 1 and 0.
    Drops duplicates and columns that have only zeros.
    
    Input
    -----
    df: Pandas DataFrame
        dataframe to be cleaned
    
    Output
    ------
    df: Pandas DataFrame
        cleaned dataframe
    """
    
    # dataframe with each category in a separate column
    categories = df.categories.str.split(pat=";",expand=True)   
    categories.columns = categories.iloc[0,:].str.split('-').str[0].tolist()
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])  
    categories.loc[categories.related==2] = 1
    
    # drop columns that have only 0s
    zero_columns = [column for column in categories if categories[column].max() == 0]
    df.drop(zero_columns, axis=1, inplace=True)
        
    # replace categories column in df with new processed category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = df.join(categories)

    df.drop_duplicates(inplace=True)
    
    return df
    
def save_data(df, database_filepath):
    """
    Saves the dataset as a table into an SQLite database.
    
    Input
    -----
    df: Pandas DataFrame
        dataset to be saved into the database
    database_filepath: string
        path of the database file 
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

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