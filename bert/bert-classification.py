import tensorflow as tf
import pandas as pd
import numpy as np
import string
import transformers
from pandas import DataFrame
from typing import Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tensorflow.python.keras.layers import Input, Dropout, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.metrics import BinaryAccuracy, Precision, Recall

MARKER_COLUMN_NAME = 'MARKER'
CLASSE_COLUMN_NAME = 'DES_CLASSE'
EMENTA_COLUMN_NAME = 'TXT_EMENTA'
METRICS = [
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall')
]


def separate_datasets(df: DataFrame, classe: string) -> Tuple[DataFrame, DataFrame]:
    print("Separando datasets")
    df_classe = df[df[CLASSE_COLUMN_NAME] == "'"+classe+"'"]
    print("Dataset da classe-alvo:", df_classe.shape)
    df_outrasClasses = df[df[CLASSE_COLUMN_NAME] != "'"+classe+"'"]
    print("Dataset das outras classes:", df_outrasClasses.shape)
    return df_classe, df_outrasClasses


def create_downsample_df(df_classe: DataFrame, df_outrasClasses: DataFrame) -> DataFrame:
    quantidadeDeRegistros = df_classe.shape[0]
    print("Criando um dataset das outras classes com ",
          quantidadeDeRegistros, " registros")
    return df_outrasClasses.sample(quantidadeDeRegistros)


def create_balanced_df(df_classe: DataFrame, df_outrasClasses: DataFrame, classeColumnName: string) -> DataFrame:
    df_outrasClasses_downsampled = create_downsample_df(
        df_classe, df_outrasClasses)
    print("Juntando os datasets...")
    df_balanced = pd.concat([df_outrasClasses_downsampled, df_classe])
    print('Verificando a quantidade de classes existentes no DataFrame após o balanceamento:')
    print(df_balanced[classeColumnName].value_counts())
    return df_balanced


def mark_classe(df_balanced: DataFrame, classe: string):
    print("Incluindo marcador na classe ", classe)
    df_balanced[MARKER_COLUMN_NAME] = df_balanced[CLASSE_COLUMN_NAME].apply(
        lambda x: 1 if x == "'"+classe+"'" else 0)
    print("Verificando marcadores:")
    print(df_balanced.sample(5))


def create_keras_model() -> Model:
    bert_preprocess = AutoTokenizer.from_pretrained(
        'neuralmind/bert-large-portuguese-cased')
    bert_encoder = AutoModel.from_pretrained(
        'neuralmind/bert-large-portuguese-cased')

    # Bert layers
    text_input = Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # Neural network layers
    l = Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = Dense(1, activation='sigmoid', name="output")(l)

    # Use inputs and outputs to construct a final model
    model = Model(inputs=[text_input], outputs=[l])
    print(model.summary)
    return model


def evaluate_model(model: Model, X_test: list):
    y_predicted = model.predict(X_test)
    y_predicted = y_predicted.flatten()
    y_predicted = np.where(y_predicted > 0.5, 1, 0)
    print(y_predicted)


def train_class(df: DataFrame, classe: string,  balanceRatio=0) -> Model:
    print('Verificando a quantidade de classes existentes no DataFrame')
    print(df[CLASSE_COLUMN_NAME].value_counts())
    df_classe, df_outrasClasses = separate_datasets(
        df, CLASSE_COLUMN_NAME, classe)
    df_balanced = create_balanced_df(df_classe, df_outrasClasses)
    mark_classe(df_balanced)
    ementas = df_balanced[EMENTA_COLUMN_NAME]
    markers = df_balanced[MARKER_COLUMN_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        ementas, markers, stratify=markers)
    model = create_keras_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=METRICS)
    model.fit(X_train, y_train, epochs=10)
    evaluate_model(model, X_test)
    nomeArquivoModelo = classe + '_bert_model.h5'
    model.save(nomeArquivoModelo)
    return model
