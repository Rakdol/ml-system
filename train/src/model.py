import pickle
from typing import Optional
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd


class HousingDataset(object):
    def __init__(
        self,
        data_directory: str,
        file_prefix: str,
        file_name: str,
        scaler_prefix: Optional[str],
        scaler_name: Optional[str],
    ):
        self.data_directory = data_directory
        self.file_prefix = file_prefix
        self.file_name = file_name
        self.scaler_prefix = scaler_prefix
        self.scaler_name = scaler_name
        self.scaler = self.get_scaler()

    def get_scaler(self):
        scaler_path = (
            Path() / self.data_directory / self.scaler_prefix / self.scaler_name
        )
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            scaler = None

        if scaler is not None:
            self.X_mean, self.X_std = scaler.mean_[:-1], scaler.scale_[:-1]
            self.n_inputs = len(scaler.mean_[:-1])
        return scaler

    def parse_csv_line(self, line):
        defs = [0.0] * self.n_inputs + [tf.constant([], dtype=tf.float32)]
        fields = tf.io.decode_csv(line, record_defaults=defs)
        return tf.stack(fields[:-1]), tf.stack(fields[-1:])

    def preprocess(self, line):
        x, y = self.parse_csv_line(line)
        return (x - self.X_mean) / self.X_std, y

    def csv_reader_dataset(
        self,
        n_readers=5,
        n_read_threads=None,
        n_parse_threads=5,
        shuffle_buffer_size=15_000,
        seed=42,
        batch_size=32,
    ):
        filepaths = str(
            Path() / self.data_directory / self.file_prefix / self.file_name
        )
        dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
        dataset = dataset.interleave(
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
            cycle_length=n_readers,
            num_parallel_calls=n_read_threads,
        )
        dataset = dataset.map(self.preprocess, num_parallel_calls=n_parse_threads)
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
        return dataset.batch(batch_size).prefetch(1)


class SimpleModel(tf.keras.Model):
    def __init__(self, n_outputs=1, **kwargs):
        super(SimpleModel, self).__init__(**kwargs)
        # self.input = tf.keras.layers.Input(shape=(n_inputs,))
        self.fc1 = tf.keras.layers.Dense(units=64, activation="relu")
        self.fc2 = tf.keras.layers.Dense(units=32, activation="relu")
        self.fc3 = tf.keras.layers.Dense(units=n_outputs, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def train():
    pass


def evaluate():
    pass
