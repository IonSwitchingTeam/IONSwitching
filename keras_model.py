#from keras import Input, Model
#from keras.layers import Embedding, Dense, Concatenate, Conv1D, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
#from keras.engine import InputSpec, Layer
#from keras import initializers
#from keras import backend as K
#from keras.layers import concatenate


from tensorflow.keras.layers import Embedding, Dense, Concatenate, Conv1D, Bidirectional, LSTM, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, SpatialDropout1D, Layer, Input, InputSpec, Attention, AdditiveAttention
from tensorflow.keras import Input, Model
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

class RCNNVariant(Model):
    def __init__(self,
                 max_len,
                 max_features,
                 embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        super(RCNNVariant, self).__init__()
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.dropout_rate = 0.5
        self.last_activation = last_activation

    def get_model(self):
        input = Input(shape=(self.max_len,))
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len, trainable=True)(input)
        embedding = SpatialDropout1D(self.dropout_rate)(embedding)
        lstm_forward = LSTM(128, return_sequences=True)(embedding)
        lstm_backward = LSTM(128, return_sequences=True, go_backwards=True)(embedding)
        x = Concatenate()([lstm_forward, embedding, lstm_backward])

        attn = AdditiveAttention()([x, x])

        x = [GlobalAveragePooling1D()(x)] + \
            [GlobalMaxPooling1D()(x)] + \
            [GlobalAveragePooling1D()(attn)] + \
            [GlobalMaxPooling1D()(attn)]

        x = Concatenate()(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model
