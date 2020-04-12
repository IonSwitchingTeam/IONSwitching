# pip install --upgrade tensorflow
# pip install keras
# pip install --user -U nltk
# pip install pandas
# pip install wordcloud

from tensorflow.keras.callbacks import EarlyStopping
#from keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing import sequence, text

from keras_model import RCNNVariant
import pandas as pd
from Preprocessor import filter_data
import numpy as np
from collections import Counter



def load_data():
    train_set1 = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
    train_set1 = filter_data(train_set1)
    train_set2 = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')
    train_set2 = filter_data(train_set2)
    train = pd.concat([train_set1, train_set2])

    val = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/validation_en.csv')
    test = pd.read_csv('jigsaw-multilingual-toxic-comment-classification/test_en.csv')
    # test = test[['id', 'content']]

    return (train.iloc[:, :], train.iloc[:, -1]), \
           (val.iloc[:, :], val.iloc[:, -1]), \
           (test.iloc[:, :], None)

def main():
    #max_features = 3000000  # maximum number of words to keep, based on word frequency
    max_features =  10000
    max_len = 500  # maximum number of words in each document
    batch_size = 32
    embedding_dims = 200
    epochs = 3

    print('Loading data...')

    (x_train, y_train), (x_val, y_val), (x_test, _) = load_data()

    x_train_text = x_train['comment_text'].values.tolist()
    x_val_text = x_val['comment_text_en'].values.tolist()
    x_test_text = x_test['content_en'].values.tolist()

    y_train = y_train.values.tolist()
    y_val = y_val.values.tolist()

    all_text = x_train_text + x_val_text + x_test_text

    #print('Number of unique words: ', len(set((' '.join(all_text)).split())))

    tokenizer = text.Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                               split=' ', char_level=False, oov_token=None)

    tokenizer.fit_on_texts(all_text)

    '''
    print('tokenizer.word_counts: ', tokenizer.word_counts)
    print('tokenizer.document_count: ', tokenizer.document_count)
    print('tokenizer.word_index: ', tokenizer.word_index)
    print('tokenizer.word_docs: ', tokenizer.word_docs)
    '''

    x_train_text = tokenizer.texts_to_sequences(x_train_text)
    x_val_text = tokenizer.texts_to_sequences(x_val_text)
    x_test_text = tokenizer.texts_to_sequences(x_test_text)

    x_train_text = sequence.pad_sequences(x_train_text, maxlen=max_len)
    x_val_text = sequence.pad_sequences(x_val_text, maxlen=max_len)
    x_test_text = sequence.pad_sequences(x_test_text, maxlen=max_len)

    model = RCNNVariant(max_len=max_len, max_features=max_features, embedding_dims=embedding_dims).get_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    model.fit(x=np.asarray(x_train_text), y=np.asarray(y_train),
              batch_size=batch_size,
              epochs=epochs)
              #callbacks=[early_stopping],
              #validation_data=(x_test, y_test))
    print(model.summary())

    print('Test...')
    result = model.predict(x_test_text)
    print('result: ', result)

if __name__ == "__main__":
    main()
