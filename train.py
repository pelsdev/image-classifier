# Imports
import argparse
import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Function for command line arguments
def arg_parser():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset of images")

    parser.add_argument('data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()

# Main function
def main():
    # Get command line arguments
    args = arg_parser()

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dir = args.data_dir + '/train'
    test_dir = args.data_dir + '/test'
    valid_dir = args.data_dir + '/valid'

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    # Load the category to name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load a pre-trained model
    if args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Unsupported architecture")
        return

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

        # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier part of the VGG model with our newly created classifier
    model.classifier = classifier

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # check if gpu available
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
        
    # Move the model to the selected device
    model.to(device)


    # Train the network
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1

            # Move inputs and labels to the device specified by args.gpu
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Keep track of the training loss
            running_loss += loss.item()

            if steps % print_every == 0:
                # Calculate validation accuracy
                accuracy = 0
                valid_loss = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        valid_loss += criterion(logps, labels).item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_loss = running_loss / len(trainloader)
                valid_loss = valid_loss / len(validloader)
                valid_accuracy = accuracy / len(validloader) * 100
                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Training loss: {train_loss:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {valid_accuracy:.2f}%")
                running_loss = 0
                model.train()
    print("Done Training ")
                
    # Save the checkpoint
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_units': args.hidden_units,
                  'dropout': 0.5,
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx,
                  'arch': args.arch}
    torch.save(checkpoint, args.save_dir)
    
main()
