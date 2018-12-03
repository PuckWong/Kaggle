def attention_3d_block(inputs, TIME_STEPS):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul
    
## intializer test ####
# gelu
def att():
    K.clear_session()       
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.2)(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True, 
                               kernel_initializer=glorot_uniform(seed=111000), 
                               recurrent_initializer=Orthogonal(gain=1.0, seed=123000)))(x)

    # 1D convolutions that can iterate over the word vectors
    #x1 = Conv1D(filters=40, kernel_size=1, padding='same', activation="relu", kernel_initializer=glorot_uniform(seed=10000))(x)
    #x2 = Conv1D(filters=40, kernel_size=2, padding='same', activation="relu", kernel_initializer=glorot_uniform(seed=20000))(x)
    #x3 = Conv1D(filters=24, kernel_size=3, padding='same', activation="relu", kernel_initializer=glorot_uniform(seed=30000))(x)
    #x4 = Conv1D(filters=24, kernel_size=5, padding='same', activation="relu", kernel_initializer=glorot_uniform(seed=40000))(x)
    attention_mul = attention_3d_block(x, maxlen)
    attention_mul = Flatten()(attention_mul)
    
    #x1 = GlobalMaxPool1D()(x1)
    #x2 = GlobalMaxPool1D()(x2)
    #x3 = GlobalMaxPool1D()(x3)
    #x4 = GlobalMaxPool1D()(x4)
    
    #c = Capsule(num_capsule=8, dim_capsule=8, routings=4, share_weights=True)(x)
    #c = Flatten()(c)
    #a = AttentionWithContext()(x)
    #concat = concatenate([x1, x2, x3, x4, c])
    #a = AttentionWeightedAverage()(concat)
    #x = Dropout(0.3)(merge1)
    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=10000))(attention_mul)
    x = Dropout(0.12)(x)###test code####
    x = BatchNormalization()(x)
    #x = Dense(100, activation="relu", kernel_initializer=glorot_uniform(seed=20000))(x)
    #x = Dropout(0.12)(x)###test code####
    #x = BatchNormalization()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=0.02),)###test code####
    return model
