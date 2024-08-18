import os
import time
import datetime
import logging
import pickle
from typing import Optional
from pathlib import Path
from io import StringIO, BytesIO
import boto3
import tensorflow as tf
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        scaler_path = str(
            Path() / self.data_directory / self.scaler_prefix / self.scaler_name
        )
        try:
            minio_client = boto3.client(
                "s3",
                endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            response = minio_client.get_object(Bucket="mlflow", Key=scaler_path)
            pkl_data = response["Body"].read()
            minio_client.close()
            scaler = pickle.load(BytesIO(pkl_data))
        except Exception as e:
            print("Exception as e", e)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

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

    def pandas_reader_dataset(
        self, target_col: str, shuffle_buffer_size=10_000, seed=42, batch_size=32
    ):
        filepaths = str(
            Path() / self.data_directory / self.file_prefix / self.file_name
        )

        try:
            minio_client = boto3.client(
                "s3",
                endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            response = minio_client.get_object(Bucket="mlflow", Key=filepaths)
            csv_data = response["Body"].read().decode("utf-8")
            minio_client.close()

            df = pd.read_csv(StringIO(csv_data))
        except Exception as e:
            print("Exception as e", e)
            df = pd.read_csv(filepaths)

        features = df.drop(labels=[target_col], axis=1).values
        feature_scaled = (features - self.X_mean) / self.X_std
        target = df[target_col].values

        dataset = tf.data.Dataset.from_tensor_slices((feature_scaled, target))
        dataset = (
            dataset.shuffle(shuffle_buffer_size, seed=seed)
            .batch(batch_size)
            .prefetch(1)
        )
        return dataset

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


def simple_model(input_shape: tuple, output_dim: int):
    model_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(16, activation="relu")(model_input)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    model_output = tf.keras.layers.Dense(output_dim, activation=None)(x)

    model = tf.keras.Model(model_input, model_output, name="simple_model")

    return model


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


@tf.function
def train_step(
    model: tf.keras.Model,
    loss_fn: tf.keras.losses,
    optimizer: tf.keras.optimizers,
    train_metric: tf.keras.metrics,
    x: tf.data.Dataset,
    y: tf.data.Dataset,
):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(
    model: tf.keras.Model,
    loss_fn: tf.keras.losses,
    test_metric: tf.keras.metrics,
    x: tf.data.Dataset,
    y: tf.data.Dataset,
):
    val_logits = model(x, training=False)
    val_losses = loss_fn(y, val_logits)
    test_metric.update_state(y, val_logits)
    return val_losses


def train(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    valid_dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers,
    loss_function: tf.keras.losses,
    train_metric: tf.keras.metrics,
    valid_metric: tf.keras.metrics,
    epochs: int = 10,
    tensorboard_directory: str = "/opt/logs/gradient_tape/",
    checkpoints_directory: str = "/opt/housing/model/",
):

    # Tensorboards
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = tensorboard_directory + current_time + "/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Batch mean metrics
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

    logger.info("start training...")
    for epoch in range(epochs):
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            losses = train_step(
                model,
                loss_function,
                optimizer,
                train_metric,
                x_batch_train,
                y_batch_train,
            )

            train_loss(losses)

            # Log every 200 batches.
            if step % 200 == 0:
                logger.info(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(losses))
                )
                logger.info("Seen so far: %d samples" % ((step + 1) * 32))

        train_mean_metric = train_metric.result()
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("rmse", train_mean_metric, step=epoch)

        logger.info(
            "Training mean metric over epoch: %.4f" % (float(train_mean_metric),)
        )
        # Reset training metrics at the end of each epoch
        train_metric.reset_states()
        train_loss.reset_states()
        logger.info("Time taken: %.2fs" % (time.time() - start_time))

        _, eval_loss = evaluate(
            model=model,
            test_dataset=valid_dataset,
            loss_function=loss_function,
            test_metric=valid_metric,
            epoch=epoch,
        )

        checkpoints = os.path.join(
            checkpoints_directory, f"epoch_{epoch}_loss_{eval_loss}"
        )
        logger.info(f"save checkpoints: {checkpoints}")
        model.save(checkpoints)


def evaluate(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    loss_function: tf.keras.losses,
    test_metric: tf.keras.metrics,
    epoch: int,
    tensorboard_directory: str = "/opt/logs/gradient_tape/",
):
    # Tensorboards
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    valid_log_dir = tensorboard_directory + current_time + "/valid"
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    # Batch mean metrics
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in test_dataset:
        val_losses = test_step(
            model, loss_function, test_metric, x_batch_val, y_batch_val
        )

    test_loss(val_losses)
    valid_mean_metric = test_metric.result()
    loss = test_loss.result()
    with valid_summary_writer.as_default():
        tf.summary.scalar("loss", test_loss.result(), step=epoch)
        tf.summary.scalar("rmse", valid_mean_metric, step=epoch)

    test_metric.reset_states()
    test_loss.reset_states()
    logging.info("Test mean metric: %.4f" % (float(valid_mean_metric),))

    return valid_mean_metric, float(loss)
