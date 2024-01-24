import os, sys
import pandas as pd

type_rationale = "consensus"
path = f"[API] CQA/{type_rationale}"

# read all files in the directory and concatenate them
files = os.listdir(path)
# df = pd.DataFrame()
df = pd.read_csv(f"[API] CQA/{type_rationale}.csv", index_col=False)

# for file in files:
#     if 'error' in file:
#         continue
#     df = pd.concat([df, pd.read_csv(f"{path}/" + file, index_col=False)])

error = pd.DataFrame()
for file in files:
    if 'error_____' in file:
        error = pd.concat([error, pd.read_csv(f"{path}/" + file, index_col=False)])

# assign rationale from error file to df
df = pd.concat([df, error[['premise', 'hypothesis', 'rationale']]])

# df = pd.concat([df, pd.read_csv("/home/huy/Desktop/HCMUS/distilling-step-by-step/[API] CQA/neutral_0.csv", index_col=False)])

df.drop_duplicates(subset=['premise', 'hypothesis'], inplace=True, keep='last')
df = df.reset_index(drop=True)
df.to_csv(f"[API] CQA/{type_rationale}.csv", index=True)