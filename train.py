import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []


for intent in intents['intents']:
# Have to use "intents" as the key because python imported it as dictionary.
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        #Extend instead of append to not get an array in arrays.
        xy.append((w, tag))
        #Used tupple

ignore_words = ["?","!",".",","]

# To check if all words have been tokenized
# print(all_words)

#Filter out ignore_words
all_words = [stem(w) for w in all_words if w not in ignore_words]
#print(all_words)

#Trick to get rid of duplicates
all_words = sorted(set(all_words))

#Probably not necessary but better to do that "youtube comment"
tags = sorted(set(tags))

#Tags doent match the video
# print(tags)

# Associated number for each tag
X_train = []
#Bag of words
y_train = []

for(pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    #Gives us numbers for our labels
    label = tags.index(tag)

    #Somtimes we want one hot encoded vektor
    # Not when using pytorch with crossentropyloss
    y_train.append(label) #CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
hidden_size = 8
# Number of tags/different classes
output_size = len(tags)

#Either leng of all words or x train 0 as they all have the same size.
input_size = len(X_train[0])
learning_rate = 0.001
#Can try out different ones
num_epochs = 1000



# Check if both have equivalen lengths, and second has the same length as all words
#print(input_size, len(all_words))
#print(output_size, tags)


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
# num_workers is for multi- processing/threading
# Set to 0 if raising an error under windows  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)


#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')


data ={
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data,FILE)

print(f'Training complete. File saved to {FILE}')