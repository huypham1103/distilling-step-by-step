import g4f
from datetime import datetime
import tqdm
import pandas as pd
import ast
from api_utils import split_answers, handle_answer
import numpy as np

class APIData:
    def __init__(self, tokens, data):
        self.tokens = tokens
        self.data = data
        self.prompt_list = []
        self.prompt_template=   '''Questions: %s, Choices: (A) %s, (B) %s, (C) %s, (D) %s, (E) %s. What is the correct answer to the question '%s' with the options %s, %s, %s, %s %s? Provide a straightforward explanation. \n'''
        self.limit = 10
        
    def get_bulk_prompt(self, prompts):
        return "Let's answer these question.\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
    
    def handle_data(self):
        self.data.rename(columns={'question': 'premise', 'choices': 'hypothesis'}, inplace=True)
        self.data['hypothesis'] = self.data['hypothesis'].apply(ast.literal_eval)
        self.data['prompt'] = self.data.apply(lambda x: self.prompt_template % (x['premise'], x['hypothesis'][0], x['hypothesis'][1], x['hypothesis'][2], x['hypothesis'][3], x['hypothesis'][4], 
                                                                                x['premise'], x['hypothesis'][0], x['hypothesis'][1], x['hypothesis'][2], x['hypothesis'][3], x['hypothesis'][4]), axis=1)
        
        self.prompts = [self.get_bulk_prompt(self.data['prompt'][i:i+10]) for i in range(0, len(self.data), 10)]
        self.premise = [self.data['premise'][i:i+10].values for i in range(0, len(self.data), 10)]
        self.hypothesis = [self.data['hypothesis'][i:i+10].values for i in range(0, len(self.data), 10)]
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
        self.data.to_csv('[API] CQA/try_another.csv', index=False)
    
    def run(self):
        self.handle_data()
        self.token = self.tokens[0]
        for i in tqdm.tqdm(range(len(self.prompts))):
            response = self.get_response(self.prompts[i])
            self.get_answer(response)
            self.match_answer(self.premise[i], self.hypothesis[i], self.list_answer)
            print(f"Finished {i+1} prompts")
    
    
data = pd.read_csv('[API] CQA/cqa_train.csv', index_col=False)[['question', 'choices']]        

tokens = [
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJtYXR0aGV3c21pdGhzd2RlakBob3RtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctZmRZeGgzM3pReXprQVVJZmlpMVRkdjltIiwidXNlcl9pZCI6InVzZXItWnFrWTFleWkycmNwd2VyMTJiSE1xeHJGIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJhdXRoMHw2NGM5OTIzODhkZDA0MmIxYjJlNGUyNTEiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzA0MTgwNDI5LCJleHAiOjE3MDUwNDQ0MjksImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIiwic2NvcGUiOiJvcGVuaWQgZW1haWwgcHJvZmlsZSBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIn0.a31rWJyiFXXaNv8MY1aK0UFGYb7MMboUZQaDlZxrMDeOachjN-al9KYt3YfG8zGkeG3ahAPjmV6ZcK_VLjDvUNO1whErQHTMjUrO-5JX9fEZeyVUgOMqu4SEo46CHVqNNsUp6RoP9tu4RAH46rea4jW7V3AVqjGn5nvocOH_cB_q3OFhtqXYr1y155zhCrhhTCjLQo1buX-ovR0ILoMgNKpvrWp5pPkJ02-iH2jNppxlBg_Fpo7YQ2ldKK_TnjxBybc-HkZ8144l9slvTpvWHL9Rm-msR5EKWZq2LfJmxhe0p-52p5Y_OGTCsDup3Tth9At4jV50x5VkXZP3DB0mSA",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJodXlxdW9jaG9hbjIwQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctb0tTZUJmbnNhck5jYlc4Mk9LWG9JUTN5IiwidXNlcl9pZCI6InVzZXIteHVTcUoxWWxKWHVDWXVZQVZFU0h2OHVkIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwODQ4NjMxNzgwODc4NzQ4MDc0NiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4Nzg5MjQsImV4cCI6MTcwNTc0MjkyNCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.NuInfdezA1TMCxm59mK9SVX0a4VlN_8RsgzHm-KRK97aio-lqUhD_ljQpxXaOjptpkCDWLeFxzVig2UTgGOw7VmgGzbaABazriqVD34KkyKN6u14S2yqdvPZKT8zLtKLiQ_njcEC2aO4ZLbXjlbZyWe1pCKCjmCGhadXiqs-v0FQqmK5RRq4a1l8FclIfDS9d9hSORZKMvLSRL1QwuVD2Xq-9P0T796T6JOexkiHqz1iPalt-76GoiDO0nS9Osa1MFYG9sQOxmgVhAOM37vJVpS0KiQ6t_SF2Qo0i1DNQY-p2wP4AW8jUjSHiDG6bEI9Pj_fDIDbbintDuUYBFHZ3Q",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJodXlxdW9jaG9hbjIxQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctbXg2bEJGU2pkY0lWV01Ndnk5eU9ZQlVKIiwidXNlcl9pZCI6InVzZXItSWFLNGVZaDBlbDltVFlYOFVINGlYM3h3In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMDM5NTUwMDMxNjQ3OTE0MjQ2NSIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4NzkwMTAsImV4cCI6MTcwNTc0MzAxMCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.iODNtgF0OBVHIZJ4xVjiB5EBo-U5sb7qe21QA0gA9AEWuctcx9x2GmkI-fsVZzFs9M6llOFKbPbtM1EZkYK2Png1tiqirYoqKoDx4s90o97N0-yx4xpJrjB6fPiR5T6N46GCEXYKOAfkcaLZ8iZiBmrZsgj249QgT2-37BFGf8OWS6e73lmN-LRX3Xrwx_zRwxWRTPueohtM2VdkPup8UzK1w64XKcvVAesH0UNHtY6T58BmYLM9nt1HRxAW_TC--oeQnZGRx965TUXVPTByr4Bpn3S8OcC1_9tsFF9FAltZW9ASqnBB60FaOAd2vP3dv09WQjo--duHk1ay1k8dKQ",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJodXlxdW9jaG9hbjIzQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctcFV4S0RrcGR3YjZZYjZrTkU3WWRzbXJtIiwidXNlcl9pZCI6InVzZXIta0EycWFIMEtON2traTAzUTZMS1AyV3N4In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNTcwMDg5OTAwNDMyNzg5ODgxNSIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4NzkwNjAsImV4cCI6MTcwNTc0MzA2MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.jhJ9tOkpLX1M2K7Os6742k75V8WKQxvksw16BS9wLfhdp7NJKDCVzDwMPziavkKmJNZMYcjUo_PGyfDmpBPQ0TswWkE8lZdxVOt_mI6aW9PxQkKQQOZ0zgliPIKOh6ggOZ0b9sJGdN73PeejmeTHNjAx9upfh5f_UmP2_B5f5ynrN3jr2qU2NX01RBq6qBgcyN0a6qVpDbfl6kYGkJBMXOnZjIB46C8WvvAX5BRe_ptTtwhIrpMbqMy0UWkYLozczKFkpCqUEFMXGkyJjKzD6eDCDB4RWn2Fxd09gH6lWITBZJ7H6Br7OMHI80-T52K55sCQ4txucd2wwJHcPHpHRA",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJodXlxdW9jaG9hbjI0QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctclI5dFlrZ09GaVFPUFVOOEJlbzZNVUY0IiwidXNlcl9pZCI6InVzZXIta0Z6SkJEM2lqNHNxb1NVT0Q2Z1gxWWpBIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwODg1MTgyNzAzNTA5OTk3NTQzNyIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4NzkxMDMsImV4cCI6MTcwNTc0MzEwMywiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.1ht-i7_V1B-MMjcxHuMEzYO0-r67E-RiEOmB4WS9vt2qm4f8cnMtxhy4eP-L0dSTUBaNmbpmppcgX3w7ta9auwYFR5b0cqbEoLym9cxTEn5Xb912dY_1kobnyNhjWOwvXVl3eVM1WxEwdSlmCQCn3ZkfmrZoRYaM5wb7H-aBtBB5Xvyp9gK_-SjIPBSlb9xXqZeOn_5bFG7RpPJATUIRG_8L_kX7ypr9Y2i54vKi9pJyke-fwWoWxBt1ZWutFaTmLdPeEWKtgUOyT41t70sbOY3sgE8jC8DXeta2iE5Bwg3sJuTmNfUkQGA-VfbvY0O-LK-AXtAwq0Nl5CI_UmVyDw",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJodXlxdW9jaG9hbjI1QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctR1M1empkVTJpdFpVNmtRSDdTdGYwR3JBIiwidXNlcl9pZCI6InVzZXItQXhVNUVERUZiUmx4amRxV3VvUGVSZnBFIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExMjIxODA4NDUwMTY2MTU5NzIyMyIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4NzkxNDIsImV4cCI6MTcwNTc0MzE0MiwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.PZnSwo2uqnbO-G_46ONp9P-2dzHS1OSfleLbK4m9NfJUHh8UxnCGLHokUCCCocoepLSlHWo26gdmHER1y32NESdQ6M_bNt8u93qhkBqg6ElvQef7ftio3qf2UnPk8jqO9crSkPhxoShiFZQ8gUaaz4fQjmMH-QmJ2n0_ClrzO8WETy9xrI1ReteDUPyb9lEHSvjQP2MsLfn0tlaZfMR72XfiKwO6Uf7hPYsFdgKtsr4KmMTX0MpbzEherZwmiU2CLDGoOdxct6rzNpyyz8HaCoOIRfQe8R_0dm1q3Efn7_-XBqOQfw8R3Wj21oB3kJR5CQpE587_wCp6GVoQY2q-4Q",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJodXlxdW9jaG9hbjI2QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctNG9LRmlFa2prTzZNSFlQaGl0WkRYU3FzIiwidXNlcl9pZCI6InVzZXItU0M4MjBucHE1NTg2MmozZlVmZ2ZPcHZlIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwNjc2MzY1OTY1Nzc2NDk5MzM0NCIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MDQ4NzkxODIsImV4cCI6MTcwNTc0MzE4MiwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUgb2ZmbGluZV9hY2Nlc3MifQ.ljkIQub11G4tumI698jjnjs-TtnbVD1MsqOudQshhfznMh4qC7GqM1cZo0P-8y_CJpUlcpCRt9cex8LK6-tIT_AZqAKx9Izsv4oEGlivrTBNuFkcSN7ebQnhgFefmdAqArh6jQ_PEbTvJ6Gmfi7YxDQnVLKlVXmUNkBEMirtnYbKGvmLtQDFdHnt_mNgNJcqgWZL4_pbgLR3Vg-Si0AZ0-VfOzZggUN82BTKY1s-It0U72hKpzLNw9d8kS79OvNpzj-IaRweUxWB5nfo0-vBmIwkQmCBnkia3JC-esx8vfcHYjnlT_-tqULw_aXcC-RlUrH_1z9ek-61IILENRUNjQ"
]

api = APIData(tokens=tokens, data=data)
api.run()
