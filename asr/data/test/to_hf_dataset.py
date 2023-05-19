import argparse
from os import listdir
from datasets import Dataset, Audio


def to_hf_dataset(
    audio_dir: str,
    save_path: str = None,
    sampling_rate: int = 16000,
    cast_audio: bool = True,
) -> Dataset:
    paths = [f"{audio_dir}/{f}" for f in listdir(audio_dir)]

    # Create Dataset
    data = {
        "path": paths,
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

    # Save Dataset
    if save_path is not None:
        dataset.save_to_disk(save_path)

    return dataset


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Path to the directory containing the audio files",
    )
    args.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the dataset to",
    )
    args = args.parse_args()

    # Create Dataset
    dataset = to_hf_dataset(
        audio_dir=args.audio_dir,
        save_path=args.save_path,
    )

    print(dataset)
