
import re
def split_answers(text):
    # Split the text using the pattern that denotes a new answer
    pattern = r'\d+\.'
    parts = re.split(pattern.strip(), text)

    # The first element will be empty, so discard it
    if parts and parts[0] == '' or len(parts[0]) < 3:
        parts.pop(0)

    # Re-add the split pattern text to each part except the first
    # for i in range(1, len(parts)):
    #     parts[i] = f"Premise:{parts[i]}"

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


if __name__ == "__main__":
    answer = """'1. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the woman is wearing a different outfit and doing a different activity than what is shown in the premise.\n\n', '2. The answer is **neutral**. The hypothesis is neither entailed nor contradicted by the premise. It is possible that the woman is showing off her car to friends, but it is not necessarily true based on the premise.\n\n', '3. The answer is **entailment**. The hypothesis is entailed by the premise, meaning that it is true based on the premise. The premise shows that the woman has a car and is in front of a house, which is what the hypothesis states.\n\n', '4. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the woman is in a different location and using a different vehicle than what is shown in the premise.\n\n', '5. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the woman is sitting on a different vehicle and waiting to get into a different place than what is shown in the premise.\n\n', "6. The answer is **entailment**. The hypothesis is entailed by the premise, meaning that it is true based on the premise. The premise shows that the woman is standing in front of Nathalie's DollHouse, which is what the hypothesis states.\n\n", "7. The answer is **neutral**. The hypothesis is neither entailed nor contradicted by the premise. It is possible that the woman is standing in front of her blue Camaro with Nathalie's DollHouse in the background, but it is not necessarily true based on the premise. The premise does not specify the model of the car or the ownership of the car.\n\n", '8. The answer is **entailment**. The hypothesis is entailed by the premise, meaning that it is true based on the premise. The premise shows that the man is on a snowboard and is snowboarding, which is what the hypothesis states.\n\n', '9. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the person on the snowboard is a woman, not a man, as shown in the premise.\n\n', '10. The answer is **neutral**. The hypothesis is neither entailed nor contradicted by the premise. It is possible that a picture is being taken of the snowboarder, but it is not necessarily true based on the premise. The premise does not show who or what is behind the camera.\n\n', '11. The answer is **neutral**. The hypothesis is neither entailed nor contradicted by the premise. It is possible that the man on the snowboard is going towards a camera because he is out of control and going to crash, but it is not necessarily true based on the premise. The premise does not show the speed or the direction of the snowboarder.\n\n', '12. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the person on the snow is on skis, not a snowboard, as shown in the premise.\n\n', '13. The answer is **entailment**. The hypothesis is entailed by the premise, meaning that it is true based on the premise. The premise shows that the man has a snowboard, which is what the hypothesis states.\n\n', '14. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the man is riding some skis, not a snowboard, as shown in the premise.\n\n', '15. The answer is **neutral**. The hypothesis is neither entailed nor contradicted by the premise. It is possible that the man is in a video about snowboarding, but it is not necessarily true based on the premise. The premise does not show the context or the purpose of the camera.\n\n', '16. The answer is **entailment**. The hypothesis is entailed by the premise, meaning that it is true based on the premise. The premise shows that the man is riding a snowboard and is approaching the camera from the top of the hill, which is what the hypothesis states.\n\n', '17. The answer is **neutral**. The hypothesis is neither entailed nor contradicted by the premise. It is possible that the man is riding a snowboard and is doing some hard tricks while traveling downhill, but it is not necessarily true based on the premise. The premise does not show the difficulty or the type of the tricks.\n\n', '18. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the man has a skateboard, not a snowboard, as shown in the premise.\n\n', '19. The answer is **entailment**. The hypothesis is entailed by the premise, meaning that it is true based on the premise. The premise shows that the man is on a snowboard and is going towards a camera, which is what the hypothesis states.\n\n', '20. The answer is **contradiction**. The hypothesis contradicts the premise by stating that the man is without a snowboard and is hiking in the snow, not snowboarding, as shown in the premise.'"""

    print(split_answers(answer))