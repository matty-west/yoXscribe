# THIS SCRIPT WILL TAKE A LONG TIME TO RUN

import pandas as pd
from pyannote.audio import Pipeline
from pyannote.core import Segment
from tqdm import tqdm
import time  

df = pd.read_csv('p1/p1.tsv', sep='\t')  

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", 
    use_auth_token="YOUR-HF-TOKEN")

import torch
pipeline.to(torch.device("cuda"))

diarization = pipeline("p1/p1_audio.wav") 

print("Diarization complete. Processing turns...") 

df['speaker'] = 'unknown'  

total_turns = len(list(diarization.itertracks(yield_label=True)))  # Get total turns
print(f"Total turns to process: {total_turns}")

with tqdm(total=total_turns, desc="Processing turns") as pbar:  # Use tqdm for progress
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_time = time.time()  # Record start time for each turn
        segment = Segment(turn.start, turn.end)

        print(f"Processing turn {i+1}/{total_turns}: {segment}")  # Print turn info

        matching_rows = df[(df['start_time'] >= segment.start) & (df['end_time'] <= segment.end)] 
        if not matching_rows.empty:
            df.loc[matching_rows.index, 'speaker'] = speaker

        end_time = time.time()  # Record end time
        print(f"Turn processed in {end_time - start_time:.2f} seconds")  # Print processing time
        pbar.update(1)  # Update progress bar

print("Finished processing turns.")

# Further analysis or save the updated DataFrame
print(df.head())  # Print the first few rows to check the results
df.to_csv('transcript_with_speakers.tsv', sep='\t', index=False)  # Save the updated DataFrame
