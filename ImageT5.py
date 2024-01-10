import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your image (replace 'path_to_your_image.jpg' with your image file path)
img = Image.open('/home/huy/Downloads/canvas.png').convert('RGB')
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Extract features
with torch.no_grad():
    features = model(batch_t)


def features_to_tokens(features, max_tokens=512, vocab_size=32100):
    # Flatten and normalize the features
    flattened = torch.flatten(features)
    normalized = (flattened - flattened.min()) / (flattened.max() - flattened.min())

    # Scale to vocabulary size and truncate/pad to max_tokens
    scaled = torch.clamp((normalized * vocab_size).int(), 0, vocab_size - 1)
    padded = torch.nn.functional.pad(scaled, (0, max_tokens - scaled.size(0)), mode='constant', value=0)
    return padded[:max_tokens]

tokens = features_to_tokens(features)


from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Convert tokens to string format
input_ids = tokens.unsqueeze(0)

# Generate text based on the tokens
output = model.generate(input_ids)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
