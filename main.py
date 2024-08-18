from argparse import ArgumentParser, RawTextHelpFormatter
import os

import mlflow


def main():

    print("Tracking URI: ", mlflow.get_tracking_uri())

    parser = ArgumentParser(
        description="Runner",
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--commit_hash",
        type=str,
        default="000000",
        help="code commit hash",
    )

    parser.add_argument(
        "--preprocess_data",
        type=str,
        default="housing",
        help="california housing",
    )
    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="./data/preprocess",
        help="preprocess downstream directory",
    )
    parser.add_argument(
        "--preprocess_cached_data_id",
        type=str,
        default="",
        help="previous run id for cache",
    )

    parser.add_argument(
        "--train_upstream",
        type=str,
        default="./preprocess/data/preprocess",
        help="upstream directory",
    )
    parser.add_argument(
        "--train_downstream",
        type=str,
        default="./train/data/model/",
        help="downstream directory",
    )
    parser.add_argument(
        "--train_tensorboard",
        type=str,
        default="./train/data/tensorboard/",
        help="tensorboard directory",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--train_learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--train_model_type",
        type=str,
        default="simple",
        help="simple",
    )

    args = parser.parse_args()
    mlflow_experiment_id = int(os.getenv("MLFLOW_EXPERIMENT_ID", 0))

    with mlflow.start_run():
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            parameters={
                "data": args.preprocess_data,
                "downstream": args.preprocess_downstream,
                "cached_data_id": args.preprocess_cached_data_id,
            },
        )
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)

        dataset = os.path.join(
            "",
            str(mlflow_experiment_id),
            preprocess_run.info.run_id,
            "artifacts/downstream_directory",
        )

        train_run = mlflow.run(
            uri="./train",
            entry_point="train",
            backend="local",
            parameters={
                "upstream": dataset,
                "downstream": args.train_downstream,
                "tensorboard": args.train_tensorboard,
                "epochs": args.train_epochs,
                "batch_size": args.train_batch_size,
                "learning_rate": args.train_learning_rate,
                "model_type": args.train_model_type,
            },
        )
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)


if __name__ == "__main__":
    main()
