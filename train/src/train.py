import argparse
import logging
import os

import mlflow
import tensorflow as tf
import tf2onnx

from src.constants import MODEL_ENUM
from src.configurations import TrainConfigurations
from src.model import simple_model, HousingDataset, train, evaluate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_run(
    mlflow_experiment_id: str,
    upstream_directory: str,
    downstream_directory: str,
    tensorboard_directory: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_type: str,
):

    train_set = HousingDataset(
        data_directory=upstream_directory,
        file_prefix=TrainConfigurations.train_prefix,
        file_name=TrainConfigurations.train_file_name,
        scaler_prefix=TrainConfigurations.scaler_prefix,
        scaler_name=TrainConfigurations.scaler_name,
    )
    train_dataset = train_set.pandas_reader_dataset(
        target_col=TrainConfigurations.target_names[0], batch_size=batch_size
    )

    valid_set = HousingDataset(
        data_directory=upstream_directory,
        file_prefix=TrainConfigurations.valid_prefix,
        file_name=TrainConfigurations.valid_file_name,
        scaler_prefix=TrainConfigurations.scaler_prefix,
        scaler_name=TrainConfigurations.scaler_name,
    )

    valid_dataset = valid_set.pandas_reader_dataset(
        target_col=TrainConfigurations.target_names[0], batch_size=batch_size
    )

    test_set = HousingDataset(
        data_directory=upstream_directory,
        file_prefix=TrainConfigurations.test_prefix,
        file_name=TrainConfigurations.test_file_name,
        scaler_prefix=TrainConfigurations.scaler_prefix,
        scaler_name=TrainConfigurations.scaler_name,
    )
    test_dataset = test_set.pandas_reader_dataset(
        target_col=TrainConfigurations.target_names[0], batch_size=batch_size
    )

    if model_type == MODEL_ENUM.SIMPLE_MODEL.value:
        n_inputs = len(TrainConfigurations.feature_names)
        model = simple_model(input_shape=(n_inputs,), output_dim=1)
    else:
        raise ValueError("Unknown model")

    mlflow.tensorflow.log_model(model, "model")

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_metric = tf.keras.metrics.RootMeanSquaredError()
    valid_metric = tf.keras.metrics.RootMeanSquaredError()

    train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        loss_function=loss_fn,
        train_metric=train_metric,
        valid_metric=valid_metric,
        epochs=epochs,
        tensorboard_directory=tensorboard_directory,
        checkpoints_directory=downstream_directory,
    )

    eval_metric, loss = evaluate(
        model=model,
        test_dataset=test_dataset,
        loss_function=loss_fn,
        test_metric=valid_metric,
        tensorboard_directory=tensorboard_directory,
        epoch=epochs + 1,
    )
    logger.info(f"Latest performance: Eval_RMSE: {eval_metric}, Loss: {loss}")

    model_file_name = os.path.join(
        downstream_directory,
        f"housing_{mlflow_experiment_id}",
    )
    onnx_file_name = os.path.join(
        downstream_directory,
        f"housing_{mlflow_experiment_id}.onnx",
    )

    model.save(model_file_name)

    spec = (tf.TensorSpec((None, 8), tf.float32, name="input"),)
    model_proto = tf2onnx.convert.from_keras(
        model, input_signature=spec, output_path=onnx_file_name
    )

    mlflow.log_param("optimizer", "RMSProp")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_metric("rmse", eval_metric)
    mlflow.log_metric("loss", loss)
    mlflow.log_artifact(model_file_name)
    mlflow.log_artifact(onnx_file_name)
    mlflow.log_artifacts(tensorboard_directory, artifact_path="tensorboard")


def main():
    parser = argparse.ArgumentParser(
        description="Train housing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--upstream",
        type=str,
        default="/opt/data/preprocess",
        help="upstream directory",
    )

    parser.add_argument(
        "--downstream",
        type=str,
        default="/opt/housing/model/",
        help="downstream directory",
    )

    parser.add_argument(
        "--tensorboard",
        type=str,
        default="/opt/housing/tensorboard/",
        help="tensorboard directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_ENUM.SIMPLE_MODEL.value,
        help="simple",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    upstream_directory = args.upstream
    # filepath = tf.keras.utils.get_file("housing_train.csv", upstream_directory)
    downstream_directory = args.downstream
    tensorboard_directory = args.tensorboard
    os.makedirs(downstream_directory, exist_ok=True)
    os.makedirs(tensorboard_directory, exist_ok=True)

    start_run(
        mlflow_experiment_id=mlflow_experiment_id,
        upstream_directory=upstream_directory,
        downstream_directory=downstream_directory,
        tensorboard_directory=tensorboard_directory,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
