import sys
import pickle
from time import time
from sqlalchemy import create_engine

import numpy as np
import pandas as pd

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(database_filepath):
    """
    Loads the data source and splits it into features X and target y.
    
    Input
    -----
    database_filepath: string
        path of disaster response database file
   
    Output
    ------
    X: numpy array
        array containing unprocessed messages
    y: Pandas DataFrame
        multi-class target dataframe containing one column per class 
   """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', con=engine)
    
    X = df['message'].values 
    y = df[df.columns[4:]]
    
    return X, y

def clean(text):
    """
    Cleans an input string. 
    Replaces all URLs with a placeholder, contracted forms with non-contracted, lowers the string and removes punctuation, digits or special characters.
    
    Input
    -----
    text: string or numeric
        text to be cleaned; in case the input is of numeric type, it is converted to string and same transformation is performed
        
    Output
    ------
    text: string
        clean text
    """
    
    text = str(text)
    
    # replace urls with string "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = text.lower()
    
    # replaces "'s" with space // example: "what's" -> "what "
    text = re.sub(r"\'s", " ", text)
    
    # replaces contracted forms with non-contracted
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    # keep only letters
    text = re.sub(r'[^a-zA-Z]', " ", text)
    
    # remove unnecessary spaces
    text = re.sub('\s+', ' ', text)
    text = text.strip()  
    
    return text

def tokenize(text):
    """
    Custom word tokenizer.
    Text is cleaned, word tokenized using nltk, custom stop words together with one letter words are removed and afterwards tokens are lemmatized.
    
    Input
    -----
    text: string
        document to be tokenized
        
    Output
    ------
    text: string
        list of tokens
    """
    
    text = clean(text)

    tokens = word_tokenize(text)
    
    # custom list of stop words consiststing of most nltk stopwords, but without negative words like 'against', 'no', 'not', considering our task
    stop_words = ["i", "me", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "can", "will", "just", "should", "now",
                 ]
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w)!=1]
    
    return tokens


def build_model():
    """
    Builds a model to be trained.
    Uses sklearn Pipeline with bag of words, tfidf transformer and then Naive Bayes classifier. Best parameters are found using GridSearch. As a metric to evaluate the model, there is used recall since the dataset is inbalanced.
    
    Output
    ------
    model: GridSearchCV
        classifier to be trained
    parameters: dictionary
        parameters to be tried during grid search
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2), lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('model_nb', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75),
        'vect__max_features': (None, 150000, 200000),
        'tfidf__use_idf': (True, False),
        'model_nb__estimator__alpha': (0.01, 0.1)
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='recall_micro')
    
    return model, parameters

def evaluate_model(model, X_test, y_test):
    """
    For a previously trained model, prints the classification report for every class.
    
    Input
    -----
    model: GridSearchCV
        trained classifier to be evaluated
    X_test: numpy array
        array containing the test set
    y_test: Pandas DataFrame
        dataframe containing labels for the test set
    """
    
    y_pred = model.predict(X_test)
    
    for i, category in enumerate(y_test):
        print(f'Class {category.upper()}:\n {classification_report(y_test.iloc[:, i].values, y_pred[:, i])}')

def save_model(model, model_filepath):
    """
    Saves a trained model in a pickle file.
    
    Input
    -----
    model: GridSearchCV
        trained classifier
    model_filepath: string
        path to the location where the model is to be saved
        should contain also the name of the pickle file
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model, parameters = build_model()
        
        print('Training model...')
        t0 = time()
        model.fit(X_train, y_train)
        print("done in %0.1fs minutes" % ((time() - t0)/60))
        
        print("Best parameters set:")
        best_parameters = model.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

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