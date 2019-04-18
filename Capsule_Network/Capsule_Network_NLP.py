#!/usr/bin/env python
# coding: utf-8

# <h2>Capsule Network on the classic IMDB Sentiment</h2>
# <p>The goal of this notebook is to explore the Capsule Network on the classic IMDB Sentiment Analysis dataset. I would like to investigate how well the CapsNet does on NLP task such as sentiment analysis and compare it to the state of the art.</p>

# In[1]:


### Import Libraries
import pandas as pd
import numpy as np
import time

import gensim
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors


# ### Get Data and Prepare it for training

# In[ ]:


### Settings to tokenize sentences and convert labels to torch floats
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)

### Get test/train split for torchtext
train, test = datasets.IMDB.splits(TEXT, LABEL)


# In[ ]:


print('Number of training examples: {}'.format(len(train)))
print('Number of training examples: {}'.format(len(test)))


# In[ ]:


### Example Review and Label
print(vars(train.examples[1]))


# In[ ]:


# build the vocabulary
TEXT.build_vocab(train)
LABEL.build_vocab(train)
vocab_dict = dict(TEXT.vocab.stoi) ###stoi = string to int


# In[ ]:


def create_wv_matrix(vocab_dict):
    print ('... Loading Word Vectors')
    word_vectors = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin", binary=True)
    wv_matrix = []
    count = 0
    
    for each in vocab_dict.items():
        count += 1
        
        word = str(each[0])
        index = int(each[1])
        
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        
        if count %10000 == 0:
            print ("On Index {}".format(count))
            
    ### Add Unknown Token
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    ### Add Pad Token
    wv_matrix.append(np.zeros(300).astype("float32"))
    print ('... Finished Creating Matrix')
    return np.array(wv_matrix)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# In[ ]:


### Create BuckerIterator 
BATCH_SIZE = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train, test), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)


# In[ ]:


### Get Word Vector Matrix
wv_matrix = create_wv_matrix(vocab_dict)


# In[ ]:


### Test the Embedding Layer
emb_layer, num_embeddings, embedding_dim = create_emb_layer(wv_matrix)
batch = next(iter(train_iterator))
emb_layer(batch.text[2]).shape


# ### Create CNN Model

# In[ ]:


class CNN(nn.Module):
    def __init__(self, weights_matrix):
        super(CNN, self).__init__()
        
        ### Embedding Layer
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        
        ### Convolution Layer 1
        self.conv1 = nn.Sequential(         # input shape (1, 300, 300)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=6,             # n_filters
                kernel_size=7,              # filter size
                stride=1,                   # filter movement/step
                padding=3,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (6, 300, 300)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # (300-2 / 2) choose max value in 2x2 area, output shape (6, 150, 150)
        )
        
        ### Convolution Layer 2
        self.conv2 = nn.Sequential(        # input shape (6, 150, 150)
            nn.Conv2d(6, 6, 7, 1, 3),     # output shape (6, 150, 150)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (6, 75, 75)
        )
            
        ### Fully Connected Layer 3
        self.FC1 = nn.Linear(6 * 75 * 75, 3000)
        
        ### Fully Connected Layer 4
        self.FC2 = nn.Linear(3000, 1000)
        
        # Output 2 classes
        self.out = nn.Linear(1000, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        output = self.out(x)
        
        return output


# ### Train CNN Model

# In[ ]:


### Hyperparameters
num_epochs = 2
cnn_classifier = CNN(weights_matrix=wv_matrix)

optimizer = optim.Adam(cnn_classifier.parameters())
criterion = nn.BCEWithLogitsLoss()


# In[ ]:


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[ ]:


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 1
    model.train()
    
    for batch in iterator:
        count +=1
        if count%5 == 0:
            print ("Batch #: {}".format(count))
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


### Training Loop
best_valid_loss = float('inf')

for epoch in range(num_epochs):

    start_time = time.time()
    
    train_loss, train_acc = train(cnn_classifier, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(cnn, test_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# In[ ]:





# In[ ]:




