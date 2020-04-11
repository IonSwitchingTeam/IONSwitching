!pip install --upgrade tensorflow
!pip install keras
!pip install pandas
!pip install wordcloud
!pip install --user -U nltk

!pip install keras.preprocessing
from keras.preprocessing import sequence, text

from google.colab import files
files.upload()
!pip install -q kaggle
!pip install transformers
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets list
!kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification

def main():
    max_features = 100000 # maximum number of words to keep, based on word frequency
    max_len = 500  # maximum number of words in each document
    batch_size = 32
    embedding_dims = 200
    epochs = 5

    print('Loading data...')

    (x_train, y_train), (x_val, y_val), (x_test, _) = load_data()

    ####################################
    ######## LOCAL SANITY CHECK ########
    ####################################

    x_train = x_train[:1000]
    x_val = x_val[:1000]
    x_test = x_test[:1000]

    y_train = y_train[:1000]
    y_val = y_val[:1000]

    x_train_text = x_train['comment_text'].values.tolist()
    x_val_text = x_val['comment_text'].values.tolist()
    x_test_text = x_test['content'].values.tolist()

    y_train = y_train.values.tolist()
    y_val = y_val.values.tolist()

    all_text = x_train_text + x_val_text + x_test_text

    print('Number of unique words: ', len(set((' '.join(all_text)).split())))

    tokenizer = text.Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                               split=' ', char_level=False, oov_token=None)

    tokenizer.fit_on_texts(all_text)

    x_train_text = tokenizer.texts_to_matrix(x_train_text)
    x_val_text = tokenizer.texts_to_matrix(x_val_text)
    x_test_text = tokenizer.texts_to_matrix(x_test_text)

    x_train_text = sequence.pad_sequences(x_train_text, maxlen=max_len)
    x_val_text = sequence.pad_sequences(x_val_text, maxlen=max_len)
    x_test_text = sequence.pad_sequences(x_test_text, maxlen=max_len)

    model = RCNNVariant(max_len=max_len, max_features=max_features, embedding_dims=embedding_dims).get_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    model.fit(x=np.asarray(x_train_text), y=np.asarray(y_train),
              batch_size=batch_size,
              epochs=epochs)
    
    print(model.summary())

    print('Test...')
    result = model.predict(x_test)
    print('result: ', result)

if __name__ == "__main__":
    main()
