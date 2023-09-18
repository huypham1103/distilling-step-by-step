import requests as r

url  = "https://api.openai.com/v1/chat/completions"

header = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.8",
    "authorization": "Bearer sess-jWYIpPba9bJxdkTZzdnX2iW00hXZFlDfwuifIWSN",
    "content-type": "application/json",
    "openai-organization": "org-fdYxh33zQyzkAUIfii1Tdv9m",
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
        "content": '''Read the questions, answers and try to pick among A,B,C,D,E.\n\n\nQuestions: "There are 10 apples on an apple tree.  Three fall off.  Now there are X apples."  What is this an example of? \n\nAnswers: (A) park, (B) coloring book, (C) garden center, (D) math problem, (E) gravity \n\nGiven the question ''',
      }
    ],
    "max_tokens": 700,
    "temperature": 1
  }

response = r.post(url, headers=header, json=body)

print(response.json())