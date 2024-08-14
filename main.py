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


if __name__ == "__main__":
    main()
