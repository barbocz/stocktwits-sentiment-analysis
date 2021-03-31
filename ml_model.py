import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import html
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from joblib import dump, load


# GridSearchCV ML model K-fold training - Multinomial Naive Bayes
def train_model_naive_bayes(training_data):
   
    # Creating a train test set for 500k labelled comments to train the model using a Moltuinomial NB classifier
    x_train, x_test, y_train, y_test = train_test_split(training_data['message'], training_data['sentiment'], 
                                                        test_size=0.2, stratify=training_data['sentiment'])

    # Create pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB())  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    # This is where we define the values for GridSearchCV to iterate over

    parameters = {

        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'classifier__alpha': (0.00001, 0.000001),
        'classifier__fit_prior': (True, False),
        # 'clf__max_iter': (10, 50, 80),
    }

    # Do 10-fold cross validation for each of the 6 possible combinations of the above params
    grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
    grid.fit(x_train, y_train)
    
    return grid, x_test, y_test


# GridSearchCV ML model K-fold training - Logistic Regression
def train_model_logistic_regression(training_data):
   
    # Creating a train test set for 500k labelled comments to train the model using a Moltuinomial NB classifier
    x_train, x_test, y_train, y_test = train_test_split(training_data['message'], training_data['sentiment'], 
                                                        test_size=0.2, stratify=training_data['sentiment'])

    # create pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', LogisticRegression())  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    # this is where we define the values for GridSearchCV to iterate over
    parameters = {

        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'classifier__C': (np.logspace(-3,3,7)),
        # 'clf__max_iter': (10, 50, 80),
    }

    # do 10-fold cross validation for each of the 6 possible combinations of the above params
    grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
    grid.fit(x_train, y_train)

    
    return grid, x_test, y_test


# Results & Classification Report
# GridSearch Results
def display_best_result(grid):
    
    print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
    print('\n')
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))


# Classification report for test set
def display_classification_report(df, grid, y_test, x_test):
   
    print('Test Set Classification Report')
    y_preds = grid.predict(x_test)
    print('accuracy score: ', accuracy_score(y_test, y_preds))
    print('\n')
    print('confusion matrix: \n', confusion_matrix(y_test, y_preds))
    print('\n')
    print(classification_report(y_test, y_preds))

    # Classification report for remaining data
    print('Remaining Data Set Classification Report')
    y_data = df["message"]
    y_preds = grid.predict(y_data)
    print('accuracy score: ', accuracy_score(df["sentiment"], y_preds))
    print('\n')
    print('confusion matrix: \n', confusion_matrix(df["sentiment"], y_preds))
    print('\n')
    print(classification_report(df["sentiment"], y_preds))


def run_model(model):
    if model == "NB":
        grid, x_test, y_test = train_model_naive_bayes(training_data)
        display_best_result(grid)
        display_classification_report(df, grid, y_test, x_test)
        return grid
    elif model == "LR":
        grid, x_test, y_test = train_model_logistic_regression(training_data)
        display_best_result(grid)
        display_classification_report(df, grid, y_test, x_test)
        dump(grid.best_estimator_, 'model\stocktwits_modelNB.pkl')
        return grid
    else:
        print('Input either:\n1. "NB" - Naive Bayes\n2. "LR" - Logistic Regression')

def remove_emoji(tweets):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweets)

def remove_stopwords(row):
    stopword_list = stopwords.words('english')
    words = []
    for word in row:
        if word not in stopword_list:
            words.append(word)
    return words

# Preprocessing Function
def tweets_preprocessing(raw_df):

    # Removing all tickers from comments
    raw_df['message'] = raw_df['message'].str.replace(r'([$][a-zA-z]{1,5})', '')

    # Make all sentences small letters
    raw_df['message'] = raw_df['message'].str.lower()

    # Converting HTML to UTF-8
    raw_df["message"] = raw_df["message"].apply(html.unescape)

    # Removing hastags, mentions, pagebreaks, handles
    # Keeping the words behind hashtags as they may provide useful information about the comments e.g. #Bullish #Lambo
    raw_df["message"] = raw_df["message"].str.replace(r'(@[^\s]+|[#]|[$])', ' ')  # Replace '@', '$' and '#...'
    raw_df["message"] = raw_df["message"].str.replace(r'(\n|\r)', ' ')  # Replace page breaks

    # Removing https, www., any links etc
    raw_df["message"] = raw_df["message"].str.replace(r'((https:|http:)[^\s]+|(www\.)[^\s]+)', ' ')

    # Removing all numbers
    raw_df["message"] = raw_df["message"].str.replace(r'[\d]', '')

    # Remove emoji
    raw_df["message"] = raw_df["message"].apply(lambda row: remove_emoji(row))

    # Tokenization
    raw_df['message'] = raw_df['message'].apply(word_tokenize)

    # Remove Stopwords
    raw_df['message'] = raw_df['message'].apply(remove_stopwords)

    # Remove Punctuation
    raw_df['message'] = raw_df['message'].apply(lambda row: [word for word in row if word not in string.punctuation])

    # Combining back to full sentences
    raw_df['message'] = raw_df['message'].apply(lambda row: ' '.join(row))

    # Remove special punctuation not in string.punctuation
    raw_df['message'] = raw_df['message'].str.replace(r"\“|\”|\‘|\’|\.\.\.|\/\/|\.\.|\.|\"|\'", '')

    # Remove all empty rows
    processed_df = raw_df[raw_df['message'].str.contains(r'^\s*$') == False]

    return processed_df


# Run
if __name__ == '__main__':
    df=pd.read_csv('data\sentiments.csv')


    df = df[["sentiment", "message"]]
    df = df[df["sentiment"].isin(["Bullish", "Bearish"])]
    bullish_df = df[df["sentiment"] == "Bullish"].sample(10000)
    bearish_df = df[df["sentiment"] == "Bearish"].sample(10000)
    training_data = pd.concat([bullish_df, bearish_df]).sample(frac=1)
    print(training_data)
    # df = pd.read_pickle("AAPL_Cleaned.pkl")

    # df = df[["sentiment", "message"]]
    # df = df[df["sentiment"].isin(["Bullish", "Bearish"])]  # Filter down into labelled comments
    #
    # # Under-sampling 30k of bullish, 30k of bearish to fix imbalance dataset
    # bullish_df = df[df["sentiment"] == "Bullish"].sample(30000)
    # bearish_df = df[df["sentiment"] == "Bearish"].sample(30000)
    # training_data = pd.concat([bullish_df, bearish_df]).sample(frac=1)
    #
    run_model("LR")



