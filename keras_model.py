from keras import Input, Model
from keras.layers import Embedding, Dense, Concatenate, Conv1D, Bidirectional, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.engine import InputSpec, Layer
from keras import initializers
from keras import backend as K
from keras.layers import concatenate


# https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/RCNNVariant/rcnn_variant.py
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
        LSTM_forward = LSTM(128, return_sequences=True)(embedding)
        LSTM_backward = LSTM(128, return_sequences=True, go_backwards=True)(embedding)
        x = Concatenate()([LSTM_forward, embedding, LSTM_backward])

        convs = []
        for i in [1, 2, 3, 4, 5]:
            conv = Conv1D(filters=128, kernel_size=i, activation='relu')(x)
            convs.append(conv)

        x = [GlobalAveragePooling1D()(conv) for conv in convs] + \
            [GlobalMaxPooling1D()(conv) for conv in convs] + \
            [AttentionWeightedAverage()(conv) for conv in convs]

        x = Concatenate()(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model

class RNN():
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        super(RCNNVariant, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

#https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
#https://arxiv.org/abs/1708.00524
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
        }
        base_config = super(AttentionWeightedAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
