import requests as r

url  = "https://api.openai.com/v1/chat/completions"

header = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.8",
    "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtYXR0aGV3c21pdGhzd2RlakBob3RtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLVpxa1kxZXlpMnJjcHdlcjEyYkhNcXhyRiJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjRjOTkyMzg4ZGQwNDJiMWIyZTRlMjUxIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5NDY3MzcxNSwiZXhwIjoxNjk1ODgzMzE1LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyJ9.DjSq30dexFS6couG-9D4J5Y22O9JlBFluQN_9rP8JVD4mpqAVVwTx5CoI3DP9vydffIOJCbPr8hJnHTjcXduUbqHHfNJ-hVf16uOfDDcfikOBvwbXFfbMJSBvmK5e5eKx3LrIsBObqvR3_9qu9LVj0efVtJCwJK4C04pxm30nrvSEqfNv5g5xnhzkd5OuoVB2_UOvLFpkpBR8lQUT59YeuuLK7XHQaYJzIgQ5b7MtCWfQc1ervG1wKXohfNLzs-LNqm8GxyU2Ue_XNOvT2i-2HIH7VG3Psc17J8ywHKip_sAiJKdv1c8gGHWBLfC7leKuWlRcoaOhvf2Dfdbnbk5aA",
    "content-type": "application/json",
    # "openai-organization": "org-fdYxh33zQyzkAUIfii1Tdv9m",
    "sec-ch-ua": "\"Not/A)Brand\";v=\"99\", \"Brave\";v=\"115\", \"Chromium\";v=\"115\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Linux\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1"
  }

body = {
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": '''Read the questions, answers and try to pick among A,B,C,D,E.\n\n\nQuestions: "There are 10 apples on an apple tree.  Three fall off.  Now there are X apples."  What is this an example of? \n\nAnswers: (A) park, (B) coloring book, (C) garden center, (D) math problem, (E) gravity \n\nGiven the question \'"There are 10 apples on an apple tree.  Three fall off.  Now there are X apples."  What is this an example of?\', which among the choices park, coloring book, garden center, math problem gravity is the correct answer? Explain your reasoning using conditional statements.''',
      }
    ],
  }

response = r.post(url, headers=header, json=body)

print(response.json())

from datasets import load_dataset
import pandas as pd

dataset = load_dataset("cos_e", 'v1.11')

train_df = pd.DataFrame(dataset['train']) 
validation = pd.DataFrame(dataset['validation'])

train_df = pd.concat([train_df, validation], ignore_index=True)

train_df = train_df[['question', 'choices']]
train_df.to_csv('train.csv', index=False)