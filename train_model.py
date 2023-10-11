#!/usr/bin/python
# encoding:utf-8
import sys
sys.setrecursionlimit(3000)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import click


def make(c_param, training_data):
    data = pd.read_csv(training_data)
    raw_data = np.array(data)
    x_data = raw_data[:, 0].reshape(-1, 1)
    y_data = raw_data[:, 1]
    lr = LogisticRegression(C=c_param)
    lr.fit(x_data, y_data)
    score = lr.score(x_data, y_data)
    mlflow.log_param("C", round(c_param, 2))
    mlflow.log_metric("score", round(score, 2))
    mlflow.sklearn.log_model(lr, "model")


@click.command()
@click.option("--c_param", "-cp", type=float, default=0.95, help="正则化系数")
@click.option("--training_data", "-td", type=str, default="", help="默认数据集")
def train(c_param, training_data):
    return make(c_param, training_data)


def eval(c_param, training_data):
    return make(c_param, training_data)


if __name__ == '__main__':
    c_param = 0.95
    training_data = './data.csv'
    eval(c_param, training_data)
