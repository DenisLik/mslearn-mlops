# Импорт библиотек

import argparse
import glob
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow


# определить функции
def main(args):
    # TO DO: включить автологирование
    mlflow.autolog()

    # читать данные
    df = get_csvs_df(args.training_data)

    # разделить данные
    X_train, X_test, y_train, y_test = split_data(df)

    # обучать модель
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Cannot use non-existent path provided: {path}"
            )
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in provided data path: {path}"
            )
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: добавить функцию для разделения данных
def split_data(df):
    X, y = df[['Pregnancies',
               'PlasmaGlucose',
               'DiastolicBloodPressure',
               'TricepsThickness',
               'SerumInsulin',
               'BMI',
               'DiabetesPedigree',
               'Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.30,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # обучаем модель
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train,
                                                             y_train)


def parse_args():
    # настройка парсера аргументов
    parser = argparse.ArgumentParser()

    # добавляем аргументы
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # парсинг аргументов
    args = parser.parse_args()

    # возврат аргументов
    return args


# запуск скрипта
if __name__ == "__main__":
    # добавление места в логи
    print("\n\n")
    print("*" * 60)

    # парсинг аргументов
    args = parse_args()

    # запуск основной функции
    main(args)

    # добавление места в логи
    print("*" * 60)
    print("\n\n")
