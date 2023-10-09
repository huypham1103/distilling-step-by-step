from datasets import load_dataset
import pandas as pd
import openai

# step 1: load dataset
dataset = load_dataset("cos_e", 'v1.11')
train_df = pd.DataFrame(dataset['train']) 
validation = pd.DataFrame(dataset['validation'])
train_df = pd.concat([train_df, validation], ignore_index=True)
train_df = train_df[['question', 'choices', 'answer']]

# step 2: define apis
apis = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
]

prompt_template= '''
                Questions: %s \n
                Answers: (A) %s, (B) %s, (C) %s, (D) %s, (E) %s \
                For the question '%s', among the choices %s, %s, %s, %s %s, which is the most likely answer? Highlight what sets your choice apart from the others.\n
                '''

def get_answer(prompt_list):
    content =  "Let's answer these question step by step. Split each question with '<end>'. \
                    which is the most likely answer? Highlight what sets your choice apart from the others. \
                    \n1.%s\n2. %s\n3. %s\n4. %s\n5. %s\n6. %s\n7. %s\n8. %s\n9. %s\n10. %s"

    content %= (
        prompt_list[0], prompt_list[1], prompt_list[2], prompt_list[3], prompt_list[4], 
        prompt_list[5], prompt_list[6], prompt_list[7], prompt_list[8], prompt_list[9],
    )
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",'content': content}
    ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    try:
        result = response['choices'][0]['message']['content']
        while 'Rate limit reached' in result:
            import random
            openai.api_key = random.choice(apis)
            response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=messages
                    )
            result = response['choices'][0]['message']['content']
        return result
    except Exception as e:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages
        )
        while 'error' in response:
            import random
            openai.api_key = random.choice(apis)
            response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=messages
                    )
        return response['choices'][0]['message']['content']