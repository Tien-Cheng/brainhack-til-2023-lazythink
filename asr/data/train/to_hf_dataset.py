import argparse
import pandas as pd
from datasets import Dataset, Audio


def to_hf_dataset(
    train_csv_path: str,
    test_size: float = 0.2,
    cast_audio: bool = True,
    sampling_rate: int = 16000,
    save_path: str = None,
) -> Dataset:
    # Read CSV
    df = pd.read_csv(train_csv_path)
    paths = df["path"].tolist()
    annotations = df["annotation"].tolist()

    # Create Dataset
    data = {
        "path": paths,
        "annotation": annotations,
    }
    if cast_audio:
        data["audio"] = paths
    dataset = Dataset.from_dict(data)
    if cast_audio:
        dataset = dataset.cast_column(
            "audio",
            Audio(
                sampling_rate=sampling_rate,
            ),
        )

    # Split Dataset
    dataset = dataset.train_test_split(
        test_size=test_size,
        shuffle=True,
    )

    # Save Dataset
    if save_path is not None:
        dataset.save_to_disk(save_path)

    return dataset


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing the training data",
        default="Train.csv",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to be used as test data",
    )
    parser.add_argument(
        "--cast_audio",
        type=bool,
        default=True,
        help="Whether to cast the audio column to an Audio object",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Sampling rate of the audio files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the dataset to",
    )
    args = parser.parse_args()

    # Create Dataset
    dataset = to_hf_dataset(
        train_csv_path=args.train_csv_path,
        test_size=args.test_size,
        cast_audio=args.cast_audio,
        sampling_rate=args.sampling_rate,
        save_path=args.save_path,
    )
    print(dataset)
