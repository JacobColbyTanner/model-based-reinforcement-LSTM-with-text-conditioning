import torch
import numpy as np  
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F



def get_action_batch(data,batch_size, block_size, num_images, train_test='train'):
    
    #loop through number of batches and get random images that are block_size long
    for i in range(batch_size):
        
        if train_test == 'train':
            #get random subject
            select = np.random.randint(0,np.round(num_images*0.75))
            action_ts = data[select]
            
        elif train_test == 'test':
            #get random subject
            select = np.random.randint(np.round(num_images*0.75),num_images)
            action_ts = data[select]

        #get random block
        block = np.random.randint(0,action_ts.shape[0]-(block_size+1))
        block_ts = action_ts[block:block+block_size,:]
        target_ts = action_ts[block+1:block+block_size+1,:]
        #append to batch
        if i == 0:
            batch = np.expand_dims(block_ts, axis=0)
            target_ts_batch = np.expand_dims(target_ts, axis=0)
        else:
            batch = np.concatenate((batch,np.expand_dims(block_ts, axis=0)),axis=0)
            target_ts_batch = np.concatenate((target_ts_batch,np.expand_dims(target_ts, axis=0)),axis=0)
    #convert to tensor
    batch = torch.tensor(batch, dtype=torch.float)
    target_ts_batch = torch.tensor(target_ts_batch, dtype=torch.float)
    return batch, target_ts_batch



@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, num_images):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_action_batch(data,batch_size, block_size, num_images, train_test=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class LSTMModel(nn.Module):
    def __init__(self, action_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.token2embedding = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(action_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, y):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #x = self.token2embedding(x)
        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        #calculate cross entropy loss between logits and y
        #loss = F.mse_loss(out, y)
        #try MAE loss
        loss = F.l1_loss(out, y)

        return out, loss

    #write code to take in a single input and generate a sequence of outputs
    def generate(self, x, max_new_tokens=2000):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        outputs = []
        
        for i in range(max_new_tokens):
            out, (h0, c0) = self.lstm(x.unsqueeze(1), (h0, c0))
            x = self.fc(out.squeeze(1))
            outputs.append(x[-1,:].unsqueeze(1))
             
        outputs = torch.cat(outputs, dim=1).T
        return outputs

        
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y, h=None):
        # Embedding the input
        x = self.embedding(x)
        if h is None:
            h = self.init_hidden(x.size(0))
        
        # Passing through RNN
        out, h = self.rnn(x, h)
        
        # Passing through the fully connected layer
        logits = self.fc(out)  # Only take the output of the last time step

        #calculate cross entropy loss between logits and y
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)

        return logits, loss

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def generate(self, x, max_new_tokens=2000, temperature=4):
        h = self.init_hidden(x.size(0)).to(x.device)
        outputs = []

        for _ in range(max_new_tokens):
            x = self.embedding(x)
            out, h = self.rnn(x, h)
            logits = self.fc(out)
            
            # Apply temperature scaling
            logits = logits.div(temperature).exp()
            
            # Sample from the distribution
            probs = F.softmax(logits[-1].squeeze(), dim=0)
            x = torch.multinomial(probs, 1)
            x = x.unsqueeze(0)
            outputs.append(x)
        
        outputs = torch.cat(outputs, dim=0)
        return outputs

