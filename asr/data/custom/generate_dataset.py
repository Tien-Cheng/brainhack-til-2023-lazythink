import os
import numpy as np
import pandas as pd

draft_file = 'temp.csv'
audio_dir = 'audio'
transcript_dir = 'transcripts'
audio_folders = [folder for folder in os.listdir(audio_dir)]
path_list, annotation_list = [], []

# Iterate through all files in the audio folder and create the temporary set
for folder in audio_folders:
    audio_files = [f'audio/{folder}_{file}' for file in os.listdir(f'{audio_dir}/{folder}')]
    path_list.append(audio_files)
    transcript_files = [transcript for transcript in os.listdir(transcript_dir)][:len(audio_files)]
    annotation_list.append(transcript_files)

# Flatten lists
path_list = list(np.concatenate(path_list).flat)
annotation_list = list(np.concatenate(annotation_list).flat)

# Output draft set
draft_dict = {
    'path': path_list,
    'annotation': annotation_list
}
df_temp = pd.DataFrame(draft_dict)
df_temp.to_csv(draft_file, index=False)