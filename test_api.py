from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro

client = Client(
    api_key="AIzaSyDG_BXLX52SyTRw31yUHHwJolymVqkc7PQ",
    provider=GeminiPro
)
response = client.chat.completions.create(
    model="gemini-pro-vision",
    messages=[{"role": "user", "content": "Classify the image with the following classes: buildings, forest, glacier, mountain, sea, street"}],
    image=open("/home/huy/Desktop/HCMUS/distilling-step-by-step/0.jpg", "rb")
)
print(response.choices[0].message.content)


# from transformers import T5Tokenizer
# import numpy as np
# from PIL import Image
# import requests

# # Assuming your image has already been processed and you have the pixel values
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# image = image.resize((224, 224))  # Resize to match model input
# pixels = np.array(image)

# # Serialize the pixel values
# # Convert the image pixels to a flat list and then to a string
# # Here, each pixel is represented by its RGB values
# pixel_values_str = ' '.join(str(value) for value in pixels.flatten())

# # Initialize T5 tokenizer
# tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Tokenize the string of pixel values
# tokens = tokenizer(pixel_values_str, return_tensors="pt")

# `tokens` is now a dictionary containing tokenized representations of your image pixel values,
# which can be fed into a T5 model (or similar) for further processing.

