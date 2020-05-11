import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Flatten,Activation
from keras.layers import Conv1D, MaxPooling1D,Dropout,LSTM, Conv2D
from keras.layers import Embedding, BatchNormalization, merge, Reshape, Lambda
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.models import model_from_json, load_model
import keras.backend as K


def create_my_ultimate_model(numerical_feature_length=8,
                             max_word_sequence_length=32,
                             max_numerical_sequence_length=8,
                             max_operator_squence_length=8,
                             hidden_1=16, hidden_2=8,
                             vocab_size=300, 
                             embedding_output_dim_1=8,
                             max_embedding_length_1=32,
                             lstm_output_dim_1=32,
                             opetator_size=8,
                             embedding_output_dim_2=1,
                             max_embedding_length_2=8,
                             operator_numerical_merge_dim=8,
                             conv_output_dim_1=16,
                             kernel_h_1=2,
                             kernel_w_1=5,
                             word_operator_numerical_merge_dim=16,
                             conv_output_dim_2=8,
                             kernel_h_2=2,
                             kernel_w_2=3,
                             hidden_3=32,
                             final_dim=5,                        
                            ):
    numerical_feature_input = Input(shape=[numerical_feature_length])  # 1
    word_sequence_input = Input(shape=[max_word_sequence_length])  # 2
    numerical_sequence_input=Input(shape=[max_numerical_sequence_length]) # 3  
    operator_sequence_input=Input(shape=[max_operator_squence_length])  # 4

    w1=Dense(hidden_1, activation='relu')(numerical_feature_input)
    w1=Dense(hidden_2, activation='relu')(w1)
    w1=BatchNormalization()(w1)

    w2=Embedding(vocab_size, embedding_output_dim_1 , input_length=max_embedding_length_1)(word_sequence_input)
    w2=Dropout(0.3)(w2)
    w2=LSTM(lstm_output_dim_1)(w2)
    w2=Dropout(0.3)(w2)

    w3=numerical_sequence_input
    w4=Embedding(opetator_size, embedding_output_dim_2 , input_length=max_embedding_length_2)(operator_sequence_input)
    w3=Reshape((-1,operator_numerical_merge_dim))(w3)
    w4=Reshape((-1,operator_numerical_merge_dim))(w4)


    #w34=K.concatenate([w3, w4], axis=-2)
    w34 = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-2))([w3,w4])
    #w34 = Lambda(k_concat_1)(w3,w4)

    w34=Reshape((-1,8,1))(w34)

    w34=Conv2D(conv_output_dim_1,kernel_size=(kernel_h_1, kernel_w_1), activation='relu')(w34)
    w34=Reshape((-1,word_operator_numerical_merge_dim))(w34)
    w2=Reshape((-1,word_operator_numerical_merge_dim))(w2)

    w234 = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-2))([w2,w34])
    #w234=K.concatenate([w2, w34], axis=-2)


    w234=Reshape((-1,8,1))(w234)
    w234=Conv2D(conv_output_dim_2,kernel_size=(kernel_h_2, kernel_w_2), activation='relu')(w234)
    w234=BatchNormalization()(w234)

    w234=Reshape((-1,1))(w234)
    w1=Reshape((-1,1))(w1)
    #w234=Flatten()(w234)


    w1234 = Lambda(lambda x: K.concatenate([x[0], x[1]],axis=-2 ))([w1,w234])
    #w1234=K.concatenate([w1, w234],axis=-1)
    w1234=Dense(hidden_3, activation='relu')(w1234)
    w1234=Dropout(0.3)(w1234)
    w1234=Dense(final_dim, activation='softmax')(w1234)


    model = Model(input=[numerical_feature_input,word_sequence_input,numerical_sequence_input,operator_sequence_input] ,output=w1234)
    return model