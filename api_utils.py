
import re
def split_answers(text):
    # Split the text using the pattern that denotes a new answer
    pattern = r'\n\d+\.\sPremise:'
    parts = re.split(pattern, text)

    # The first element will be empty, so discard it
    if parts and parts[0] == '':
        parts.pop(0)

    # Re-add the split pattern text to each part except the first
    for i in range(1, len(parts)):
        parts[i] = f"Premise:{parts[i]}"

    return parts

def handle_answer(answer):
    answers = []
    for i in range(1, 11):
        start_idx = answer.find(str(i))
        if i != 10:
            end_idx = answer.find(str(i + 1))
            answers.append(answer[start_idx:end_idx])
        else:
            answers.append(answer[start_idx:])
    return answers