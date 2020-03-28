import warnings
warnings.filterwarnings("ignore")
import re
import numpy as np
import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS

TEXT_COLUMN = 'comment_text'
TOXIC_COLUMN = 'toxic'

def clean_training_dataframe(df):
    df[TEXT_COLUMN] = remove_domain_specific_information(df[TEXT_COLUMN])
    df = filter_data(df)
    df = lemmatize_text(df)
    df = remove_stop_words(df)
    df = convert_toxic_column_to_boolean(df)
    df = remove_punctuation(df)
    return df

def remove_domain_specific_information(text):
    text = text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub('\\n',' ',str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)",'',str(x)))
    return text

def remove_stop_words(df):
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
    return df

def remove_punctuation(df):
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df

def lemmatize_text(df):
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: ' '.join([lem.lemmatize(item) for item in x.split()]))
    return df

def filter_data(df):
    cols_filter = ['id', 'comment_text', 'toxic']
    df = df[cols_filter]
    return df

def convert_toxic_column_to_boolean(df):
    df[TOXIC_COLUMN] = df[TOXIC_COLUMN] > 0.5
    df[TOXIC_COLUMN].value_counts(normalize=True)
    return df


#val["comment_text"] = remove_domain_specific_information(val["comment_text"])
#test["content"] = remove_domain_specific_information(test["content"])
stopwords=set(STOPWORDS)
lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
np.random.seed(0)

val = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('jigsaw-toxic-comment-train.csv')
cleaned_df = clean_training_dataframe(train)
print(cleaned_df['comment_text'])



