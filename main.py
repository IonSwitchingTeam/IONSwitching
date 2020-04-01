# pip install --upgrade tensorflow
# pip install keras

from keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from keras.preprocessing import sequence

from keras_model import RCNNVariant

def main():
    max_features = 90000 #maximum number of words in training set
    max_len = 200 #maximum number of words in each document
    batch_size = 32
    embedding_dims = 200
    epochs = 5

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)...')
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = RCNNVariant(max_len=max_len, max_features=max_features, embedding_dims=embedding_dims).get_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    print('Train...')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              #callbacks=[early_stopping],
              validation_data=(x_test, y_test))
    print(model.summary())

    print('Test...')
    result = model.predict(x_test)
    print('result: ', result)



if __name__ == "__main__":
    main()
