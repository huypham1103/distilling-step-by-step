import pandas as pd
import numpy as np
import os
# read data

def handle_round_bracket(line):
    return line[1].split(')')[0] if len(line) == 2 else np.nan



df = pd.read_csv('/home/huy/Desktop/HCMUS/distilling-step-by-step/datasets/cqa/if-else/rationales_data.csv')

# split ( char
df['answer'] = df['final_reasons'].str.split('answer is ').apply(handle_round_bracket)

print(df.head())