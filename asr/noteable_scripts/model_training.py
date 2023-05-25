# Import libraries
import os
import csv
import copy
import json
import math
import numpy as np
import pandas as pd

# Model Training
import nemo.collections.asr as nemo_asr
# NeMo's Experiment manager to handle checkpoint saving and logging
from nemo.utils import exp_manager
from omegaconf import OmegaConf, open_dict 
from pytorch_lightning import Trainer, seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Model Inferencing
import soundfile as sf
from pyloudnorm import Meter

# Reproducibility
seed_everything(42, workers=True)

# Variables
FREEZE_ENCODER = True
VOCAB_SIZE = 32000
TOKENIZER = os.path.join("tokenizers", f"tokenizer_spe_bpe_v{VOCAB_SIZE}")
TOKENIZER_TYPE_CFG = "bpe"
AUGMENT_AUDIO = False
EPOCHS = 10
BATCH_SIZE = 100

# Manifest files location
TRAIN_DATA_DIR = 'data/train/'
TEST_DATA_DIR = 'data/test/'
FILE_TEST = 'data/test/SampleSubmission_Advanced.csv'
train_manifest_file = os.path.join(TRAIN_DATA_DIR, 'train_manifest.json')
val_manifest_file = os.path.join(TRAIN_DATA_DIR, 'val_manifest.json')
test_manifest_file = os.path.join(TEST_DATA_DIR, 'test_manifest.json')
# Load file for storing model predictions
test_set = pd.read_csv(FILE_TEST)

'''
Prepare configurationnfor conformer transducer
'''
# Load pretrained model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name='stt_en_fastconformer_transducer_large')

# Enable batch normalization and squeeze excite
def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

# Convert manifest files to json format
def manifest_tojson(manifest_file: str) -> str:
    manifest_list = []
    with open(manifest_file, "r") as file:
        for line in file:
            record = json.loads(line)
            manifest_list.append(record)
    return manifest_list

# TIL output is purely uppercase alphabet and space so we normalize the output to that
def normalize_to_til(transcript: str) -> str:
    result = "".join([c.upper() if c.isalpha() else " " for c in transcript])
    # Remove double spaces
    while "  " in result:
        result = result.replace("  ", " ")
    return result

# Get predictions from trained model
def asr_predict(manifest_file, start_idx, end_idx):
    predicted_transcripts = []
    paths = []
    count = 0
    for record in manifest_file[start_idx:end_idx]:
        transcript, metadata = asr_model.transcribe([record['audio_filepath']])
        pred = normalize_to_til(transcript[0])
        # Attempt to increase loudness if unable to transcribe
        if pred == "":
            count +=1
            path = record['audio_filepath']
            data, rate = sf.read(path)
            meter = Meter(rate)
            # measure the loudness
            loudness = meter.integrated_loudness(data)
            print("Loudness:", loudness)
            for target_loudness in (-24, -14, -11, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0):
                print("Trying target:", target_loudness)
                # calculate the gain needed to normalize
                gain = target_loudness - loudness
                # apply gain correction
                normalized_data = data * 10**(gain/20.0)
                # write out the normalized audio
                sf.write("normalized_data.wav", normalized_data, rate)
                transcript, metadata = asr_model.transcribe(["normalized_data.wav"])
                pred = normalize_to_til(transcript[0])
                if pred != "":
                    break
            print("After normalisation:", pred)
        predicted_transcripts.append(pred)
        paths.append(record['audio_filepath'][16:])
    print("Number of audio files too soft:", count)

    return predicted_transcripts, paths

# Create csv file for submission
def create_csv(paths, predictions, output_filename):
    # Ensure that paths and predictions are of the same length
    assert len(paths) == len(predictions), "Lists must have the same length"

    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['path', 'annotation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for path, prediction in zip(paths, predictions):
            writer.writerow({'path': path, 'annotation': prediction})

# Freeze encoder, since the TIL dataset is quite small, it is better to freeze the encoder
if FREEZE_ENCODER:
    asr_model.encoder.freeze()
    asr_model.encoder.apply(enable_bn_se)
    print("Model encoder has been frozen")
else:
    asr_model.encoder.unfreeze()
    print("Model encoder has been un-frozen")


# Update model configuration
cfg = copy.deepcopy(asr_model.cfg)

# Set up tokenizer
cfg.tokenizer.dir = TOKENIZER
cfg.tokenizer.type = TOKENIZER_TYPE_CFG

# Set tokenizer config
asr_model.cfg.tokenizer = cfg.tokenizer

# Set up data loaders
# Setup train, validation, test configs
with open_dict(cfg):    
    # Train dataset
    cfg.train_ds.manifest_filepath = train_manifest_file
    cfg.train_ds.normalize_transcripts = False
    cfg.train_ds.batch_size = 32
    cfg.train_ds.num_workers = 8
    cfg.train_ds.pin_memory = True
    cfg.train_ds.trim_silence = True

    # Validation dataset
    cfg.validation_ds.manifest_filepath = val_manifest_file
    cfg.validation_ds.normalize_transcripts = False
    cfg.validation_ds.batch_size = 8
    cfg.validation_ds.num_workers = 8
    cfg.validation_ds.pin_memory = True
    cfg.validation_ds.trim_silence = True

    # Test dataset
    cfg.test_ds.manifest_filepath = test_manifest_file
    cfg.test_ds.normalize_transcripts = False
    cfg.test_ds.batch_size = 8
    cfg.test_ds.num_workers = 8
    cfg.test_ds.pin_memory = True
    cfg.test_ds.trim_silence = True

# Set up data loaders with new configs
asr_model.setup_training_data(cfg.train_ds)
asr_model.setup_validation_data(cfg.validation_ds)
asr_model.setup_test_data(cfg.test_ds)

# Set up optimizer and scheduler
with open_dict(asr_model.cfg.optim):
    asr_model.cfg.optim.lr = 5e-3 # from model config
    asr_model.cfg.optim.betas = [0.9, 0.98]  # from model config
    asr_model.cfg.optim.weight_decay = 1e-3  # Original weight decay
    asr_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
    asr_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
    asr_model.cfg.optim.sched.min_lr = 5e-4  # from model config

# Fused Batch
# Two lines to enable the fused batch step
asr_model.cfg.joint.fuse_loss_wer = True
asr_model.cfg.joint.fused_batch_size = 16
asr_model.cfg.compute_eval_loss = True

# Audio augmentation
with open_dict(asr_model.cfg.spec_augment):
    if AUGMENT_AUDIO:
        asr_model.cfg.spec_augment.freq_masks = 2
        asr_model.cfg.spec_augment.freq_width = 25
        asr_model.cfg.spec_augment.time_masks = 2
        asr_model.cfg.spec_augment.time_width = 0.05
    else:
        asr_model.cfg.spec_augment.freq_masks = 0
        asr_model.cfg.spec_augment.time_masks = 0

'''
Setup Trainer and Experiment Manager
'''
if torch.cuda.is_available():
    accelerator = 'gpu'
else:
    accelerator = 'cpu'

# Initialize a Trainer for the Transducer model
trainer = Trainer(devices=1,
                  accelerator=accelerator,
                  max_epochs=EPOCHS,
                  enable_checkpointing=False,
                  logger=False,
                  log_every_n_steps=5,
                  check_val_every_n_epoch=5,
                  deterministic=True)

# Setup model with the trainer
asr_model.set_trainer(trainer)

# Update model's internal configuration
asr_model.cfg = asr_model._cfg

# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
os.environ.pop('NEMO_EXPM_VERSION', None)

exp_config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/',
    name=f"nvidia-stt-ft-02-spe",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

exp_config = OmegaConf.structured(exp_config)

logdir = exp_manager.exp_manager(trainer, exp_config)

# Train model
trainer.fit(asr_model)

'''
Evaluate Model
'''
# Evaluate model on the validation set
trainer.test(asr_model)

TEST_MANIFEST = manifest_tojson(test_manifest_file)

start_idx = 0
end_idx = BATCH_SIZE
iterations = math.ceil(len(TEST_MANIFEST) / BATCH_SIZE)
all_transcripts = []
all_paths = []
asr_model.to("cuda") # Use cuda
for batch in range(iterations):
    print(f'Batch {batch+1}: Records {start_idx} to {end_idx}')
    predicted_transcripts, paths = asr_predict(TEST_MANIFEST, start_idx, end_idx)
    all_transcripts.extend(predicted_transcripts)
    all_paths.extend(paths)
    start_idx += BATCH_SIZE
    end_idx += BATCH_SIZE

# Combine all lists of transcripts into a single list to update the annotation column
test_set['annotation'] = list(np.concatenate(all_transcripts).flat)
# Output the submission csv file
test_set.to_csv('eval_submission_nvidia_stt.csv', index=False)

create_csv(all_paths, all_transcripts, 'eval_submission_nvidia_conformer_transducer_ft_02_normalised.csv')