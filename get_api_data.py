import g4f
from datetime import datetime
import tqdm
import pandas as pd
import ast
from api_utils import split_answers, handle_answer
import numpy as np
import warnings
import asyncio
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

class APIData:
    def __init__(self, tokens, data, token_idx, idx):
        self.tokens = tokens
        self.data = data
        self.prompt_list = []
        self.prompt_template=   '''Questions: %s, Choices: (A) %s, (B) %s, (C) %s, (D) %s, (E) %s. What is the correct answer to the question '%s' with the options %s, %s, %s, %s %s? Provide a straightforward explanation. \n'''
        self.limit = 10
        self.token_idx = token_idx
        self.idx = idx
        
    def get_bulk_prompt(self, prompts):
        return "Let's answer these question.\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
    
    def handle_data(self):
        self.data.rename(columns={'question': 'premise', 'choices': 'hypothesis'}, inplace=True)
        self.data['hypothesis'] = self.data['hypothesis'].apply(ast.literal_eval)
        self.data['prompt'] = self.data.apply(lambda x: self.prompt_template % (x['premise'], x['hypothesis'][0], x['hypothesis'][1], x['hypothesis'][2], x['hypothesis'][3], x['hypothesis'][4], 
                                                                                x['premise'], x['hypothesis'][0], x['hypothesis'][1], x['hypothesis'][2], x['hypothesis'][3], x['hypothesis'][4]), axis=1)
        
        self.prompts = [self.get_bulk_prompt(self.data['prompt'][i:i+self.limit]) for i in range(0, len(self.data), self.limit)]
        self.premise = [self.data['premise'][i:i+self.limit].values for i in range(0, len(self.data), self.limit)]
        self.hypothesis = [self.data['hypothesis'][i:i+self.limit].values for i in range(0, len(self.data), self.limit)]
        self.data['rationale'] = np.nan
    
    def call_api(self, messages, token):
        return g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            provider=g4f.Provider.OpenaiChat,
            stream=False,
            auth="token",
            access_token=token
            # ignore_stream_and_auth=True
        )
    
    def get_response(self, prompt):
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",'content': prompt}
        ]
        try:
            return self.call_api(messages, self.token)
        except Exception as e:
            print('**Exception**')
            result = ''
            while not result:
                try:
                    result = self.call_api(messages, self.token)
                except Exception as e:
                    print(e)
                    print(self.token)
            return result
    
    def get_answer(self, response):
        try:
            if '<end>\n' in response:
                list_answer = response.split('<end>\n')
            else:
                list_answer = response.split('\n\n') if '\n\n' in  response else response.split('\n')

            if len(list_answer) > self.limit:
                list_answer = response.split('\n\n') if '\n\n' in  response else response.split('\n')

            if len(list_answer) > self.limit:
                list_answer = response.split('\n\n\n')

            if len(list_answer) > self.limit:
                list_answer = split_answers(response)

            if len(list_answer) != self.limit:
                raise
        except Exception:
            list_answer = handle_answer(response)
            
        self.list_answer = list_answer
    
    def match_answer(self, premise, hypothesis, answer):
        index_mask = self.data['premise'].isin(premise) & self.data['hypothesis'].isin(hypothesis)
        self.data.loc[index_mask, 'rationale'] = answer
        self.data.to_csv(f'[API] CQA/neutral_{self.idx}.csv', index=False)
    
    def run(self):
        self.handle_data()
        self.token = self.tokens[self.token_idx]
        for i in tqdm.tqdm(range(len(self.prompts))):
            response = self.get_response(self.prompts[i])
            self.get_answer(response)
            self.match_answer(self.premise[i], self.hypothesis[i], self.list_answer)
            # print(f"Finished {i+1} prompts")


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i in range(0, 10):
            start = i * 1000
            end = (i+1) * 1000
            data = pd.read_csv('[API] CQA/cqa_train.csv', index_col=False)[['question', 'choices']][start:end]

            tokens = [
                "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtYXR0aGV3c21pdGhzd2RlakBob3RtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctZmRZeGgzM3pReXprQVVJZmlpMVRkdjltIiwidXNlcl9pZCI6InVzZXItWnFrWTFleWkycmNwd2VyMTJiSE1xeHJGIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2NGM5OTIzODhkZDA0MmIxYjJlNGUyNTEiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzA0MTgwNDI5LCJleHAiOjE3MDUwNDQ0MjksImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.a31rWJyiFXXaNv8MY1aK0UFGYb7MMboUZQaDlZxrMDeOachjN-al9KYt3YfG8zGkeG3ahAPjmV6ZcK_VLjDvUNO1whErQHTMjUrO-5JX9fEZeyVUgOMqu4SEo46CHVqNNsUp6RoP9tu4RAH46rea4jW7V3AVqjGn5nvocOH_cB_q3OFhtqXYr1y155zhCrhhTCjLQo1buX-ovR0ILoMgNKpvrWp5pPkJ02-iH2jNppxlBg_Fpo7YQ2ldKK_TnjxBybc-HkZ8144l9slvTpvWHL9Rm-msR5EKWZq2LfJmxhe0p-52p5Y_OGTCsDup3Tth9At4jV50x5VkXZP3DB0mSA",
                "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtYXR0aGV3c21pdGhzd2RlakBob3RtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctZmRZeGgzM3pReXprQVVJZmlpMVRkdjltIiwidXNlcl9pZCI6InVzZXItWnFrWTFleWkycmNwd2VyMTJiSE1xeHJGIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2NGM5OTIzODhkZDA0MmIxYjJlNGUyNTEiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzA0MTExOTE5LCJleHAiOjE3MDQ5NzU5MTksImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.CutH5iAUe4MsP1jG42dH0xarN_Way3_XMifKLV5pZO0xW0ajWdgcygT4A3Q_DjtaxyYzlxy2w7pE2NoSsNxfOrPWPD88tC7xL6BkS6OHjgbFTa9P9Ob02P9iF4fFrJpUjgebfI5ypycIe6-2u44sW2imTLkDs_N1CKBMWe1-cMf9RBFv0fvUMMOP3FOyuAugOHT4EaR09uV49SMm_zKvsKPr4xrXTT5VVJv6zuOtopYaZlpgpOmAb08_li5XQBhic_AzgsjbdON8OSDnBhOvG-RWBAx1kk7SA_SzmTyWXQGfMhy03myzbF26q9byAQIfmXChSPwKe8h8GM-PFzPCig",
                "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJ0YW5ndGh1eWxpbmgxMTA0QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctTnZsTnJBbElSQ0IycDFTQ0xzR29DOEJ0IiwidXNlcl9pZCI6InVzZXItdEdRVmxkaHpuUzlOdEhqc2RsYWRDV3h6In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExNzMyMDI0MjQzMjMzMjY5MzIwMSIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4OTM0NTcsImV4cCI6MTcwNTc1NzQ1NywiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.k0mrNhDOHVk6MDraKBdJlGAkGWW7pNBHz7RsY_Tqp0U7fiaQA2-bLgdbl1laAScL6xjWv-qShIxxIwuxoCNYw5mlHtduAgL4PUJaOlyTEFkP9SAni0asIQ9vZmD0IL_f10cwbSF0InzmxQAz-rYsg6Tc1IBXkzykV1Q6nGrXAm1GclfAHVySGNEjQdzzv_TBi36DcBOJX6SC2S0mmAUntw7MvAf8JcIsOVUOwqswLjyPEiRplnW_uKnEYUnxxS58hoFPip7hHomF6GGq-9_2t15fQPSn0Ov1BDUYZPOiPrp-k2VS0Qtp-4zDuXR-o-ggQT-kkZiuu2EtjOI94Lk56A"
            ]
            # run get api data parallel by 3 tokens
            # print(f"Start api at token {i}")
            executor.submit(APIData(tokens, data, i%3, i).run)
