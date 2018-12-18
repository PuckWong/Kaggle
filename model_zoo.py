def singleGRU():
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.22, seed=1024)(x)
    x = Bidirectional(CuDNNGRU(112, return_sequences=True, 
                                kernel_initializer=glorot_normal(seed=12300), 
                               recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

    x = GlobalMaxPool1D()(x)
    x = Dense(1, activation="sigmoid",kernel_initializer=glorot_normal(seed=12300))(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.02),)
    return model
    
    
def singleGRU_II():
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.22, seed=1024)(x)
    x, x_h, x_c = Bidirectional(CuDNNGRU(128, return_sequences=True, return_state=True,
                                kernel_initializer=glorot_normal(seed=12300), 
                               recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

    x1 = GlobalMaxPool1D()(x)
    x2 = GlobalAvgPool1D()(x)
    x3 = Lambda(lambda x:K.max(x, axis=-1))(x)
    c = concatenate([x1, x2, x3], axis=-1)
    x = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=12300))(c)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.08),)
    return model
    
    def GRU_Attention():
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.22, seed=1024)(x)
    x = Bidirectional(CuDNNGRU(136, return_sequences=True, 
                                kernel_initializer=glorot_normal(seed=12300), 
                               recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

    x = AttentionWeightedAverage()(x)
    x = Dense(1, activation="sigmoid",kernel_initializer=glorot_normal(seed=12300))(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.02),)
    return model
    
   #epoch=5
def parallelRNN():
    K.clear_session()
    recurrent_units = 128
    inp = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features,
                                embed_size,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)(inp)
    embedding_layer = SpatialDropout1D(0.2, seed=1024)(embedding_layer)

    x = Bidirectional(CuDNNGRU(60, return_sequences=True, 
                                   kernel_initializer=glorot_uniform(seed=125422), 
                                   recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(embedding_layer)
    y = Bidirectional(CuDNNLSTM(60, return_sequences=True,
                                  kernel_initializer=glorot_uniform(seed=111000), 
                                  recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(embedding_layer)
    c = concatenate([x, y], axis=2)

    #last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)
    #x = Conv1D(filters=72, kernel_size=2, padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=10000))(x)
    #y = Conv1D(filters=72, kernel_size=2, padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=101000))(y)
    #a = Multiply()([x, y])
    #c = AttentionWithContext()(c)
    c = GlobalMaxPooling1D()(c)
    #c = BatchNormalization()(c) 
    #c = concatenate([x, y])

    output_layer = Dense(1, activation="sigmoid", kernel_initializer=glorot_uniform(seed=111000))(c)
    model = Model(inputs=inp, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.06))
    return model 
    
#epoch=7
def poolrnn():
    K.clear_session()
    inp = Input(shape=(maxlen,))
    embedding_layer = Embedding(max_features,
                                embed_size,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)(inp)
    embedding_layer = SpatialDropout1D(0.22, seed=1024)(embedding_layer)

    rnn_1 = Bidirectional(CuDNNGRU(112, return_sequences=True, 
                                   kernel_initializer=glorot_uniform(seed=111000), 
                                   recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(embedding_layer)

    #last = Lambda(lambda t: t[:, -1], name='last')(rnn_1)
    maxpool = GlobalMaxPooling1D()(rnn_1)
    #attn = AttentionWeightedAverage()(rnn_1)
    average = GlobalAveragePooling1D()(rnn_1)

    c = concatenate([maxpool, average], axis=1)
    #c = Reshape((4, -1))(c)
    #c = Lambda(lambda x:K.sum(x, axis=1))(c)
    #x = BatchNormalization()(c)
    #c = GlobalMaxPooling1D()(c)
    #x = Dense(100, activation='relu', kernel_initializer=glorot_uniform(seed=111000),)(x)
    #x = Dropout(0.12)(x)
    #x = BatchNormalization()(x)
    #x = Dense(100, activation="relu", kernel_initializer=glorot_uniform(seed=111000),)(x)
    #x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    output_layer = Dense(1, activation="sigmoid", kernel_initializer=glorot_uniform(seed=111000))(c)
    model = Model(inputs=inp, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.02))
    return model
    
    
def BiLSTM_CNN():
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.22, seed=2048)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True, 
                               kernel_initializer=glorot_normal(seed=111000), 
                               recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(x)

    x1 = Conv1D(filters=96, kernel_size=1, padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=110000))(x)
    x2 = Conv1D(filters=72, kernel_size=2, padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=120000))(x)
    x3 = Conv1D(filters=48, kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=130000))(x)
    x4 = Conv1D(filters=24, kernel_size=5, padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=140000))(x)

    x1 = GlobalMaxPool1D()(x1)
    x2 = GlobalMaxPool1D()(x2)
    x3 = GlobalMaxPool1D()(x3)
    x4 = GlobalMaxPool1D()(x4)

    c = concatenate([x1, x2, x3, x4])

    x = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=110000))(c)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.02))
    return model
    
def doubleRNN():
    K.clear_session()       
    x_input = Input(shape=(maxlen,))
    
    emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False, name='Embedding')(x_input)
    emb = SpatialDropout1D(0.22, seed=11110000)(emb)

    rnn1 = Bidirectional(CuDNNGRU(72, return_sequences=True, kernel_initializer=glorot_uniform(seed=111100), 
                           recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(emb)
    rnn2 = Bidirectional(CuDNNGRU(72, return_sequences=True, kernel_initializer=glorot_uniform(seed=111000), 
                           recurrent_initializer=Orthogonal(gain=1.0, seed=1203000)))(rnn1)

    x = concatenate([rnn1, rnn2])
    x = GlobalMaxPooling1D()(x)  
    x_output = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=111100))(x)
    model = Model(inputs=x_input, outputs=x_output)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.08),)
    return model
    
def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
        
        
        
        
def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x) # New
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model  
    
    
def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def model_gru_atten_3(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
