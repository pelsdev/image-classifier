import argparse
import torch
from PIL import Image
import numpy as np
import json
from torch import nn
from torchvision import models, transforms

# Define command line arguments
parser = argparse.ArgumentParser(description='Predict flower name from an image with predict.py along with the probability of that name.')
parser.add_argument('image_path', help='The path to the image')
parser.add_argument('checkpoint', help='The checkpoint file to use for predicting the image class')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Load the category to name mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
checkpoint = torch.load(args.checkpoint)

# Rebuild the model
model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(4096, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Use GPU if available
device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
model.to(device)

# Predict the class of the image


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    # Define transforms for the input image
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open the image and apply the transforms
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image_transforms(image).float()

    # Return the processed image as a PyTorch tensor
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Process the image
    img = process_image(image_path)
    img = img.unsqueeze_(0)

    # Move the image to the device
    img = img.to(device)

    # Turn off gradient tracking
    with torch.no_grad():
        # Get the output of the model
        output = model(img)

        # Get the probabilities and class indices
        ps = torch.exp(output)
        probs, indices = ps.topk(topk)

        # Convert indices to class labels
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in indices.cpu().numpy()[0]]

    # Convert class labels to
    labels = []
    for class_idx in classes:
        labels.append(cat_to_name[str(class_idx)])

    return probs.cpu().numpy()[0], labels


probs, classes = predict(args.image_path, model, args.top_k)

probs = [p * 100 for p in probs]
# Print the results
print("Top", args.top_k, "predictions for the image", args.image_path, ":")
for i in range(args.top_k):
    print("Class:", classes[i], "Probability:", round(probs[i], 2), "%")