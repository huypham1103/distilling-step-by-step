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

# g4f.debug.logging = True  # Enable debug logging
# g4f.debug.version_check = False  # Disable automatic version checking
# print(g4f.Provider.Bing.params)  # Print supported args for Bing

FOLDER = 'if_else'
TYPE = 'error_'

prompt_template=   {
    'consensus': '''Premise: %s, Hypothesis: %s. What is the commonly agreed-upon answer to the premise '%s' and hypothesis %s with options "entailment", "neutral", "contradiction"? Justify your answer based on general knowledge. \n''',
    'if_else': '''Premise: %s, Hypothesis: %s. Given the premise '%s' and hypothesis %s, which among the choices "entailment", "neutral", "contradiction", is the correct answer? Explain your reasoning using conditional statements.\n''',
    'contrastive': '''Premise: %s, Hypothesis: %s. For the premise '%s' and hypothesis %s, among the choices "entailment", "neutral", "contradiction", which is the most likely answer? Highlight what sets your choice apart from the others.\n''',
    'neutral': '''Premise: %s, Hypothesis: %s. What is the correct answer to the premise '%s' and hypothesis %s with the options "entailment", "neutral", "contradiction"? Provide a straightforward explanation. \n''',
    'causal': '''Premise: %s, Hypothesis: %s. For the premise '%s' and hypothesis %s, what is the correct answer among the options "entailment", "neutral", "contradiction"? Explain the cause-and-effect relationship between the question and your chosen answer.\n''',
    'comparative': '''Premise: %s, Hypothesis: %s. Compare the options "entailment", "neutral", "contradiction" and identify the most likely answer to the premise '%s' and hypothesis %s. Explain why your choice is better than the others.\n''',
    'historical': '''Premise: %s, Hypothesis: %s. Based on what is historically known, what is the most likely answer to the premise '%s' and hypothesis %s with options "entailment", "neutral", "contradiction"? Provide historical context in your rationale.\n''',
}

class APIData:
    def __init__(self, data, token_idx, idx, tokens=None):
        self.tokens = tokens
        self.data = data
        self.prompt_list = []
        self.prompt_template =   prompt_template[f'{FOLDER}']
        self.limit = 10
        self.token_idx = token_idx
        self.idx = idx
        
    def get_bulk_prompt(self, prompts):
        return "Let's answer these question.\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
    
    def handle_data(self):
        self.data.rename(columns={'question': 'premise', 'choices': 'hypothesis'}, inplace=True)
        # self.data['hypothesis'] = self.data['hypothesis'].apply(ast.literal_eval)
        self.data['prompt'] = self.data.apply(lambda x: self.prompt_template % (x['premise'], x['hypothesis'], 
                                                                                x['premise'], x['hypothesis']), axis=1)
        
        self.prompts = [self.get_bulk_prompt(self.data['prompt'][i:i+self.limit]) for i in range(0, len(self.data), self.limit)]
        self.premise = [self.data['premise'][i:i+self.limit].values for i in range(0, len(self.data), self.limit)]
        self.hypothesis = [self.data['hypothesis'][i:i+self.limit].values for i in range(0, len(self.data), self.limit)]
        self.data['rationale'] = np.nan
    
    def call_api(self, messages, token):
        return g4f.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            provider=g4f.Provider.Bing,
            stream=False,
            # timeout=120,
            # auth="token",
            # access_token=token
            # ignore_stream_and_auth=True
        )
    
    def get_response(self, prompt):
        messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
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
                    # print(e)
                    # print(self.token)
                    pass
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
        try:
            index_mask = self.data['premise'].isin(premise) & self.data['hypothesis'].isin(hypothesis)
            self.data.loc[index_mask, 'rationale'] = answer[:len(premise)]
            self.data.to_csv(f'[API] ESNLI/{FOLDER}/{TYPE}_{self.idx}.csv', index=False)
        except Exception as e:
            print(answer)
            self.data.to_csv(f'[API] ESNLI/{FOLDER}/{TYPE}_{self.idx}.csv', index=False)
        
    def run(self):
        self.handle_data()
        # self.token = self.tokens[self.token_idx]
        self.token = None
        for i in tqdm.tqdm(range(len(self.prompts))):
            response = self.get_response(self.prompts[i])
            self.get_answer(response)
            self.match_answer(self.premise[i], self.hypothesis[i], self.list_answer)
            # print(f"Finished {i+1} prompts")


if __name__ == '__main__':
    # data = pd.read_csv('[API] ESNLI/paper.csv', index_col=False)[['premise', 'hypothesis']]
    data = pd.read_csv('[API] ESNLI/error.csv', index_col=False)[['premise', 'hypothesis']]

    with ThreadPoolExecutor(max_workers=20) as executor:
        for i in range(0, 100):
            start = i * 100
            end = (i+1) * 100
            sub_data = data[start:end]

            # run get api data parallel by 3 tokens
            # print(f"Start api at token {i}")
            # APIData(data, i%3, i, tokens=None).run()
            executor.submit(APIData(sub_data, i%3, i, tokens=None).run)
            # print(start, end)
            # break

# import g4f

# g4f.debug.logging = True  # Enable debug logging
# g4f.debug.version_check = False  # Disable automatic version checking
# print(g4f.Provider.Bing.params)  # Print supported args for Bing

# # Using automatic a provider for the given model
# ## Streamed completion
# response = g4f.ChatCompletion.create(
#     model=g4f.models.gpt_4,
#     messages=[{"role": "user", "content": "Hello"}],
#     stream=True,
#     provider=g4f.Provider.Bing
# )
# for message in response:
#     print(message, flush=True, end='')

## Normal response
# response = g4f.ChatCompletion.create(
#     model=g4f.models.gpt_4,
#     messages=[{"role": "user", "content": "Hello"}],
# )  # Alternative model setting

# print(response)