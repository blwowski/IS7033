
# coding: utf-8

# <h2>Capsule Network on the classic IMDB Sentiment</h2>
# <p>The goal of this notebook is to explore the Capsule Network on the classic IMDB Sentiment Analysis dataset. I would like to investigate how well the CapsNet does on NLP task such as sentiment analysis and compare it to the state of the art.</p>

# In[17]:


### Import Libraries
import pandas as pd
import numpy as np
import time
import gc
import re

import gensim
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils


from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors


# ### Get Data and Prepare it for training

# In[18]:


df = pd.read_csv('./data/imdb_master.csv', encoding="latin-1")
df = df.drop(['Unnamed: 0', 'file'], axis=1)

df = df[df['label'] != 'unsup']
df['label'] = df['label'].replace('neg', 0)
df['label'] = df['label'].replace('pos', 1)


# In[19]:


df['label'] = df['label'].replace('neg', 0)
df['label'] = df['label'].replace('pos', 1)


# In[20]:


### Contractions Dictionary
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": " he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"i'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"We're": "we are",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who has",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))


# In[21]:


### function to clean the comment_text data
def expand_contractions(s, contractions_dict=contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def clean_data(text):
    text = expand_contractions(text, contractions)
    text = str(text).lower()
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' url ', text)
    text = re.sub('(!+)', '!', text)
    text = re.sub('(\?+)', '?', text)
    text = re.sub('(\s+)', ' ', text)
    text = re.sub('(\"+)', ' \" ', text)
    text = re.sub('(\.+)', '\.', text)
    text = re.sub('(<+)', ' < .', text)
    text = re.sub('(>+)', ' > .', text)
    text = text.replace("\\", " ")
    text = text.replace("-", " ")
    text = text.replace("—", " ")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = re.sub('([.,!?()])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('[^A-Za-z0-9\?\!\.<>,\s]+', '', text)
    return text


# In[22]:


def tokenize_pad_slice_column_to_int(df, row_name, vocab_dict, max_sequence_length):
    processed_row = []
    
    for i, row in df.iterrows():
        text = clean_data(str(row[row_name]))
        text_arr = text.split(" ")
        
        if len(text_arr) > max_sequence_length:
            text_arr = text_arr[:max_sequence_length]
        elif len(text_arr) < max_sequence_length:
            pad_to_add = max_sequence_length-len(text_arr)
            text_arr.extend(['<pad>' for i in range(pad_to_add)])
        
        for i in range(len(text_arr)):
            if text_arr[i] in vocab_dict:
                text_arr[i] = vocab_dict[text_arr[i]]
            else:
                text_arr[i] = vocab_dict[text_arr[1]]
            
        processed_row.append(np.array(text_arr))
        
    df["processed_row"] = processed_row
    return df

def create_vocab_from_df_column(df, row_name):
    vocab_dict = {'<pad>' : 0,
                  '<unk>' : 1}
    index = 2
    
    for i, row in df.iterrows():
        text = clean_data(str(row[row_name]))
        text_arr = text.split(" ")
        for word in text_arr:
            if word in vocab_dict:
                continue
            else:
                vocab_dict[word] = index
                index += 1
                
    return vocab_dict
                


# In[23]:


vocab_dict = create_vocab_from_df_column(df, 'review')


# In[24]:


df = tokenize_pad_slice_column_to_int(df, 'review', vocab_dict, 300)


# In[25]:


df.head()


# In[26]:


### Create batch generator
df_train = df[df['type'] == 'train'].sample(frac=1).reset_index(drop=True).drop(['type'], axis = 1)
df_train.label = df_train.label.astype('float64')

df_test = df[df['type'] == 'test'].sample(frac=1).reset_index(drop=True).drop(['type'], axis = 1)
df_test.label = df_test.label.astype('float64')


# In[27]:


X_train = df_train.processed_row.tolist()
X_train = torch.from_numpy(np.vstack(X_train))

X_test = df_test.processed_row.tolist()
X_test = torch.from_numpy(np.vstack(X_test))

y_train = df_train.label.tolist()
#y_train = torch.from_numpy(np.vstack(y_train), dtype=torch.long)
y_train = torch.tensor(np.vstack(y_train), dtype=torch.float)

y_test = df_test.label.to_list()
#y_test = torch.from_numpy(np.vstack(y_test))
y_test = torch.tensor(np.vstack(y_train), dtype=torch.float)

print("Train Input Shape: {} Train Target Shape: {}".format(X_train.shape, y_train.shape))
print("Test Input Shape: {} Test Target Shape: {}".format(X_test.shape, y_test.shape))


# In[28]:


train = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)


# In[29]:


test = data_utils.TensorDataset(X_test, y_test)
test_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)


# In[33]:


def create_wv_matrix(vocab_dict):
    print ('... Loading Word Vectors')
    word_vectors = KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin", binary=True)
    wv_matrix = []
    count = 0
    print ('... Finish Loading Word Vectors')
    
    for each in vocab_dict.items():
        count += 1
        
        word = str(each[0]).lower()
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
    
    del(word_vectors)
    
    return np.array(wv_matrix)

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# In[34]:


### Get Word Vector Matrix
wv_matrix = create_wv_matrix(vocab_dict)


# In[36]:


gc.collect()


# In[37]:


### Test the Embedding Layer
emb_layer, num_embeddings, embedding_dim = create_emb_layer(wv_matrix)
batch = next(iter(train_loader))
emb_layer(batch[0]).shape


# ### Create CNN Model

# In[38]:


class CNN(nn.Module):
    def __init__(self, weights_matrix):
        super(CNN, self).__init__()
        
        ### Embedding Layer
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
       
        ### Convolution Layer 1
        self.conv1 = nn.Sequential(         # input shape (1, 300, 300)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=3,             # n_filters
                kernel_size=10,             # filter size
                stride=2,                   # filter movement/step
                padding=0,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (3, 146, 146)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # (146-2 / 2) choose max value in 2x2 area, output shape (3, 73, 73)
        )
        
        ### Convolution Layer 2
        self.conv2 = nn.Sequential(        # input shape (3, 73, 73)
            nn.Conv2d(3, 3, 5, 2, 1),      # output shape (3, 36, 36)
            nn.ReLU(),                     # activation
            nn.MaxPool2d(2),               # output shape (3, 18, 18)
        )
            
        ### Fully Connected Layer 3
        self.FC1 = nn.Linear(3 * 18 * 18, 800)
        
        ### Fully Connected Layer 4
        self.FC2 = nn.Linear(800, 200)
        
        # Output 2 classes
        self.out = nn.Linear(200, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        #print (x.shape)
        x = x.unsqueeze(1)
        #print (x.shape)
        x = self.conv1(x)
        #print (x.shape)
        x = self.conv2(x)
        #print (x.shape)
        x = x.view(x.size(0), -1)           # flatten the output of conv2
        x = F.relu(self.FC1(x))
        #print (x.shape)
        x = F.relu(self.FC2(x))
        #print (x.shape)
        output =  torch.sigmoid(self.out(x))
        #print (output.shape)
        
        return output


# ### Train CNN Model

# In[39]:


### Hyperparameters
num_epochs = 50
cnn_classifier = CNN(weights_matrix=wv_matrix)

optimizer = optim.Adam(cnn_classifier.parameters())
criterion = nn.MSELoss()


# In[40]:


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[41]:


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    count = 1
    model.train()
    
    for batch in iterator:
        count +=1
        optimizer.zero_grad()
        predictions = model(batch[0])
        loss = criterion(predictions, batch[1])
        acc = binary_accuracy(predictions, batch[1])
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[42]:


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch[0])
            loss = criterion(predictions, batch[1])
            acc = binary_accuracy(predictions, batch[1])
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[43]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


### Training Loop
best_test_loss = float('inf')

for epoch in range(num_epochs):

    start_time = time.time()
    
    train_loss, train_acc = train(cnn_classifier, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(cnn_classifier, test_loader, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(cnn_classifier.state_dict(), 'tut4-model.pt')
    
    print("EPOCH: {}, TIME: {}:{}".format(epoch, epoch_mins, epoch_secs))
    print("Train Loss: {}. Train Acc: {}".format(train_loss, train_acc*100))
    print("Test Loss: {}. Test Acc: {}".format(test_loss, test_acc*100))

