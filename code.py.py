#!/usr/bin/env python
# coding: utf-8

# In[79]:


import math
import torchtext
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader , TensorDataset
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import io
import time
import pandas as pd
import numpy as np
import pickle
# import sentencepiece as spm
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# print(torch.cuda.get_device_name(0))
# device = 'cpu'
print(device)
import nltk 
nltk.download('punkt')
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer('basic_english')


# In[80]:


# Log in to your W&B account
import wandb
wandb.login(key="788422b2647292758f3ffd151b58fd95bb8ce3fc")


# In[ ]:





# In[81]:


#load data
with open('/kaggle/input/dataset/train.en', 'r') as f:
    train_en = f.readlines()

# convert to pandas 

train_en = pd.DataFrame(train_en)
with open('/kaggle/input/dataset/train.fr', 'r') as f:
    train_fr = f.readlines()
train_fr = pd.DataFrame(train_fr)

with open('/kaggle/input/dataset/test.en', 'r') as f:
    test_en = f.readlines()
test_en = pd.DataFrame(test_en)

with open('/kaggle/input/dataset/test.fr', 'r') as f:
    test_fr = f.readlines()
test_fr = pd.DataFrame(test_fr)

with open('/kaggle/input/dataset/dev.en', 'r') as f:
    dev_en = f.readlines()
dev_en = pd.DataFrame(dev_en)

with open('/kaggle/input/dataset/dev.fr', 'r') as f:
    dev_fr = f.readlines()
dev_fr = pd.DataFrame(dev_fr)


# In[82]:


tokenizer = get_tokenizer('basic_english')

# tokenize the data

train_en_tokenized = [tokenizer(i) for i in train_en[0]]
train_en_tokenized[0]
train_fr_tokenized = [tokenizer(i) for i in train_fr[0]]
train_fr_tokenized[0]
test_en_tokenized = [tokenizer(i) for i in test_en[0]]
test_en_tokenized[0]
test_fr_tokenized = [tokenizer(i) for i in test_fr[0]]
test_fr_tokenized[0]
dev_en_tokenized = [tokenizer(i) for i in dev_en[0]]
dev_fr_tokenized = [tokenizer(i) for i in dev_fr[0]]
dev_fr_tokenized[0]
dev_en_tokenized[0]


# In[83]:


from torch.utils.data import Dataset, DataLoader
#Creatin vocabulary
def create_vocab(tokenized):
    vocab = {}
    freq = {}
    #add <PAD> and <UNK> tokens
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<EOS>'] = 2
    vocab['<SOS>'] = 3
    freq['<PAD>'] = 0
    freq['<UNK>'] = 0
    freq['<EOS>'] = 0
    freq['<SOS>'] = 0
    #add tokens from tokenized sentences to vocab and freq
    for sent in tokenized:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
                freq[word] = 1
            else:
                freq[word] += 1
            
    #words with freq less than 2 are replaced with <UNK> token
    vocab_final = {}
    vocab_final['<PAD>'] = 0
    vocab_final['<UNK>'] = 1
    vocab_final['<EOS>'] = 2
    vocab_final['<SOS>'] = 3
    #add tokens from tokenized sentences to vocab_final if freq is greater than 2
    for word in vocab:
        if freq[word] >= 2:
            vocab_final[word] = len(vocab_final)
    return vocab_final


# In[84]:


#build vocab from tokenized sentences
vocab_en = create_vocab(train_en_tokenized)
print(list(vocab_en.items())[:10])
#print length of vocab
print(len(vocab_en))

vocab_fr = create_vocab(train_fr_tokenized)
print(list(vocab_fr.items())[:10])
#print length of vocab
print(len(vocab_fr))


# In[85]:


#Function to pad sentences to max length
def pad_sents(sents, pad_token, max_len):
    padded_sents = []
    for sent in sents:
        if len(sent) < max_len-2:
            padded_sents.append(['<SOS>']+sent +["<EOS>"]+ [pad_token] * ((max_len-2) - len(sent)))
        else:
            padded_sents.append(sent[:max_len])
    return padded_sents


# In[86]:


padded_train_en = pad_sents(train_en_tokenized, '<PAD>', 250)
padded_train_fr = pad_sents(train_fr_tokenized, '<PAD>', 250)
padded_test_en = pad_sents(test_en_tokenized, '<PAD>', 250)
padded_test_fr = pad_sents(test_fr_tokenized, '<PAD>', 250)
padded_dev_en = pad_sents(dev_en_tokenized, '<PAD>', 250)
padded_dev_fr = pad_sents(dev_fr_tokenized, '<PAD>', 250)


# In[87]:


#Changing tokens in tokenized sentences to indices
def token2index_dataset(tokenized):
    indices = []
    for sent in tokenized:
        index = []
        for word in sent:
            if word in vocab_en:
                index.append(vocab_en[word])
            else:
                index.append(vocab_en['<UNK>'])
        indices.append(index)
    return indices
#Changing tokens in tokenized sentences to indices
def token2index_dataset_fr(tokenized):
    indices = []
    for sent in tokenized:
        index = []
        for word in sent:
            if word in vocab_fr:
                index.append(vocab_fr[word])
            else:
                index.append(vocab_fr['<UNK>'])
        indices.append(index)
    return indices


# In[88]:


# functionn to give random encoding of words of dimension n 


# In[89]:


train_en_indices = token2index_dataset(padded_train_en)
train_fr_indices = token2index_dataset_fr(padded_train_fr)
test_en_indices = token2index_dataset(padded_test_en)
test_fr_indices = token2index_dataset_fr(padded_test_fr)
dev_en_indices = token2index_dataset(padded_dev_en)
dev_fr_indices = token2index_dataset_fr(padded_dev_fr)



# In[90]:


dev_fr_indices[0][15:]


# In[91]:


#dataloader class
class dataloader_encoder(Dataset):
    def __init__(self, en):
        self.en = en
        # self.fr = fr
        
    def __len__(self):
        return len(self.en)
    
    def __getitem__(self, idx):
        en_sent = torch.tensor(self.en[idx])
        # fr_sent = torch.tensor(self.fr[idx])
        return en_sent


# In[92]:


#dataloader class
class dataloader_encoder_fr(Dataset):
    def __init__(self, fr):
        self.fr = fr
        # self.fr = fr
        
    def __len__(self):
        return len(self.fr)
    
    def __getitem__(self, idx):
        fr_sent = torch.tensor(self.fr[idx])
        # fr_sent = torch.tensor(self.fr[idx])
        return fr_sent


# In[93]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):

        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):

        out = self.embed(x).to(device)
        return out


# In[94]:


train_dataset_en = dataloader_encoder(train_en_indices)
test_dataset_en  = dataloader_encoder(test_en_indices)
(train_dataset_en[0]).shape


# In[95]:


train_dataset_fr = dataloader_encoder_fr(train_fr_indices)
test_dataset_fr  = dataloader_encoder_fr(test_fr_indices)
(train_dataset_fr[0]).shape


# In[96]:


train_dataset_en = torch.tensor(train_en_indices).to(device)
test_dataset_en  = torch.tensor(test_en_indices).to(device)
# dev_dataset_en = torch.tensor(dev_en_indices)
(train_dataset_en).shape


# In[97]:


train_dataset_fr = torch.tensor(train_fr_indices).to(device)
test_dataset_fr  = torch.tensor(test_fr_indices).to(device)
(train_dataset_fr).shape


# In[98]:


# initialise_random_word_embeddings = Embedding(len(vocab_en), 512)
# initialise_random_word_embeddings.to(device)


# In[99]:


train_dataset_en[0]
# initialise_random_word_embeddings(train_dataset_en[0].to(device)).shape


# In[100]:


# initialise_random_word_embeddings_fr = Embedding(len(vocab_fr), 512)
# initialise_random_word_embeddings_fr.to(device)
# initialise_random_word_embeddings_fr(train_dataset_fr[0].to(device)).shape


# In[101]:


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat).to(device) #(32,10,512) -> (32,10,512)
       
        return output


# In[102]:


# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.


class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):

        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):

      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        x=x.to(device)
        return x
               


# In[103]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):

        attention_out = self.attention(key,query,value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)).to(device) #32x10x512

        return norm2_out



class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out).to(device)

        return out  #32x10x512


# In[104]:


# t = TransformerEncoder(250, len(vocab_en), 512)


# In[105]:


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
        
    
    def forward(self, key, query, x,mask):

        #we need to pass mask mask only to fst attention
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(key, query, value).to(device)

        
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) 
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        
        x = self.word_embedding(x)  #32x10x512
        x = self.position_embedding(x) #32x10x512
        x = self.dropout(x)
     
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask) 

        out = F.softmax(self.fc_out(x)).to(device)

        return out


# In[106]:


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length,num_layers=2, expansion_factor=4, n_heads=8):
        super(Transformer, self).__init__()
        
        
        self.target_vocab_size = target_vocab_size

        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    
    def make_trg_mask(self, trg):

        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask    

    def decode(self,src,trg):

        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size,seq_len = src.shape[0],src.shape[1]
        #outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len): #10
            out = self.decoder(out,enc_out,trg_mask) #bs x seq_len x vocab_dim
            # taking the last token
            out = out[:,-1,:]
     
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out,axis=0)
          
        
        return out_labels
    
    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg).to(device)
        enc_out = self.encoder(src).to(device)
   
        outputs = self.decoder(trg, enc_out, trg_mask).to(device)
        return outputs




# In[107]:


# print(train_dataset_en.shape,train_dataset_fr.shape)
model = Transformer(embed_dim=512, src_vocab_size=len(vocab_en), 
                    target_vocab_size=len(vocab_fr), seq_length=250,
                    num_layers=2, expansion_factor=4, n_heads=8)

model = model.to(device)


# In[108]:


# wrtie training loop with wandb
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


# In[109]:


# out = model(train_dataset_en, train_dataset_fr).to(device)
# out.shape


# In[110]:


train_dataset_en_loaded = DataLoader(train_dataset_en, batch_size=8, shuffle=True)
train_dataset_fr_loaded = DataLoader(train_dataset_fr, batch_size=8, shuffle=True)
test_dataset_en_loaded = DataLoader(test_dataset_en, batch_size=8, shuffle=True)
test_dataset_fr_loaded = DataLoader(test_dataset_fr, batch_size=8, shuffle=True)
# dev_dataset_en_loaded = DataLoader(dev_dataset_en, batch_size=8, shuffle=True)
# dev_dataset_fr_loaded = DataLoader(dev_dataset_fr, batch_size=8, shuffle=True)


# In[ ]:





# In[121]:


import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4

def bleu_score(targets, predictions):
    score = 1
    for i in range(len(targets)):
        target_sentence = str(targets[i])
        prediction_sentence = str(predictions[i])
        score += sentence_bleu([target_sentence], prediction_sentence, smoothing_function=smoothie)
    return score/len(targets)


# In[ ]:


# Log in to your W&B account
get_ipython().system('pip install wandb -qU')
import wandb
wandb.login(key="788422b2647292758f3ffd151b58fd95bb8ce3fc")
wandb.init(
    project="transformer",
    name="transformer",
    config={
        "learning_rate": 0.001,
        "batch_size": 8,  # Corrected to lowercase
        "epochs": 5,      # Corrected to lowercase
        "embedding_dim": 512,
        "num_layers": 2
    }
)

epochs = 5

combined_dataset = TensorDataset(train_dataset_en, train_dataset_fr)

# Create a DataLoader for batching
batch_size = 8
dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

dataloader
bleu = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for src, trg in dataloader:
        optimizer.zero_grad()

        # Forward pass
        src = src.to(device)
        trg = trg.to(device)
        outputs = model(src, trg).to(device)

        # Calculate loss
        trg = trg.view(-1)  # Flatten target sequence
        loss = criterion(outputs.view(-1, len(vocab_fr)), trg)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=2)
        predictions = predictions.view(-1).cpu().numpy()
        targets = trg.cpu().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        # accuracy = accuracy_score(all_targets, all_predictions)
        # f1 = f1_score(all_targets, all_predictions, average="weighted")
        
#         print("Bleu Score: ",bleu_score(all_targets, all_predictions))
        bleu.extend(bleu_score(all_targets, all_predictions))
        

    wandb.log({"epoch_loss": total_loss, "epoch_accuracy": accuracy, "epoch_f1": f1})
    # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

#     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f} , Accuracy: {accuracy_score(all_targets, all_predictions):.4f} , F1 score: {f1_score(all_targets, all_predictions, average='macro'):.4f}")


# Save the trained model
# torch.save(model.state_dict(), "transformer_model.pth")
wandb.finish()

    




# In[ ]:





# In[21]:


# print bleu scores in a file 
with open('bleu_train.txt', 'w') as f:
    for item in bleu:
        f.write("%s\n" % item)

average_bleu = sum(bleu)/len(bleu)
print("Average Bleu Score For Train Data" ,average_bleu)


# In[ ]:





# In[24]:


test_dataset_en = torch.tensor(test_en_indices)
test_dataset_fr  = torch.tensor(test_fr_indices)
combined_dataset = TensorDataset(test_dataset_en, test_dataset_fr)
batch_size = 8
dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
model.eval()
all_predictions = []
all_targets = []
bleu = []
for src, trg in dataloader:
    outputs = model(src, trg)
    all_predictions.extend(outputs.argmax(-1).tolist())
    all_targets.extend(trg.tolist())
# print(f"Accuracy: {accuracy_score(all_targets, all_predictions):.4f} , F1 score: {f1_score(all_targets, all_predictions, average='macro'):.4f}")
print("Bleu Score: ",bleu_score(all_targets, all_predictions))
bleu.extend(bleu_score(all_targets, all_predictions))

# # print bleu scores in a file 

with open('bleu_test.txt', 'w') as f:
    for item in bleu:
        f.write("%s\n" % item)

average_bleu = sum(bleu)/len(bleu)
print("Average Bleu Score For test Data" ,average_bleu)



# In[ ]:


train_dataset_en.shape ,train_dataset_fr.shape


# In[ ]:


# for i , j in dataloader:
#     print(i , j);


# In[ ]:




