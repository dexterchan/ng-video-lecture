#%%
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import requests


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 #reduce learning way by 10 ; bring down learning rate when u have more parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'# for mac users with m1/m2 chips
eval_iters = 200
N_EMBD = 384
N_HEAD = 6
N_LAYERS = 6
DROP_OUT = 0.2

#%% ------------

torch.manual_seed(1337)
if not os.path.exists('input.txt'):
    #download https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt to input.txt
    requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', allow_redirects=True, stream=True)
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
start_time = datetime.now()
print(f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

#declare no backpropagating function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #set model in evaluation mode where dropout disabled
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #set model back to training mode where dropout activated
    return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(DROP_OUT)

    def forward(self, x):
        """
            Cross attention notes:
                x is private information of the tokens
                v is the projected information of the single head after query dot key
                for cross attention
                key and value would come from another source (not x!)
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # next, we gather historical information
        # every token emits two information vectors: query and key
        # query vector: what am I looking for?
        # key vector: what do I contain?
        # the dot product of query and key becomes weight here
        # if query and key is highly correlated, the weight has high value
        # if query and key is NOT correlated, the weight becomes very small -> 0.00
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        # v is the projected information of the single head after query dot key
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROP_OUT), # Add dropout with 0.4 probability
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        # declare feed forward
        self.ffwd = FeedForward(n_embd)
        # declare layer norm 1 #https://arxiv.org/pdf/1607.06450
        self.ln1 = nn.LayerNorm(n_embd)
        # declare layer norm 2 # https://arxiv.org/pdf/1607.06450
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):

        # Deep residual learning to counter deep nested - https://arxiv.org/pdf/1512.03385
        # feed to the self attention head. by one head
        # reminder "+=" not working here
        x = x + self.sa(self.ln1(x)) # apply multi head of attention (B, T, N_EMBD)
        
        # feed forward to add more thinking here before feeding to linear layer for final logit convention
        x = x + self.ffwd(self.ln2(x)) # (B, T, N_EMBD)
        return x

# # implement layerNorm here
# class BatchNorm1d:
#     def __init__(self, dim, eps=1e-5, momentum=0.1):
#         self.eps = eps
#         self.gamma = torch.ones(dim)
#         self.beta = torch.zeros(dim)

#     def set_device(self, device):
#         self.gamma = self.gamma.to(device)
#         self.beta = self.beta.to(device)

#     def __call__(self, x):
#         # calculate the forward pass
#         xmean = x.mean(1, keepdim=True) # batch mean
#         xvar = x.var(1, keepdim=True) # batch variance
#         xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
#         self.out = self.gamma * xhat + self.beta
#         return self.out

#     def parameters(self):
#         return [self.gamma, self.beta]

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed, n_layers):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # declare a positional embedding
        # For look period of T, we have n_embed vector for each time instance
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=N_HEAD) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        # declare a linear layer to project the embedding to the vocab size
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # B = Batch Size ~ 32 running 32 batch in parallel
        # T = Block size ~ 8 running looking back 8 samples
        # C = Class ~ vocab size here
        token_embed = self.token_embedding_table(idx) # (B,T,N_EMBD)
        # get position embedding from 0 to T-1
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, N_EMBD)
        x = token_embed + pos_embed # (B, T, N_EMBD) + (T, N_EMBD) = (B, T, N_EMBD) broadcasted

        # replaced by multiple blocks
        # #feed to the self attention head. by one head
        # #x = self.sa_head(x)
        # x = self.sa_heads(x) # apply multu head of attention (B, T, N_EMBD)

        # # feed forward to add more thinking here before feeding to linear layer for final logit convention
        # x = self.ffwd(x) # (B, T, N_EMBD)
        x = self.blocks(x)

        x = self.ln_f(x) # (B, T, N_EMBD)


        # x is not just the token embedding of the meaning but also contain the temporal information as well
        # convert the token embedding to logits
        # if use linear alone, not enough thought process (distinctive power) to convert token embedding into logits
        logits = self.lm_head(x) #B, T, C(Vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_cond = idx[:, -block_size:] #limit the input to be block size of T
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
#%%
model = BigramLanguageModel(vocab_size, N_EMBD, N_LAYERS)
m = model.to(device)
xb, yb = get_batch('train')
out, loss = m(xb, yb)
# Print tensor shapes
print(f"logits shape: {out[0].shape}")  
#output size is batch_size * Block size , vocabsize
print(f"loss: {loss}") # expect loss to be -1 * ln(1/65)
#%%
outputs = m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=20)
print(outputs.shape)
print(decode( outputs[0].tolist()))

#%%
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model = model.to(device)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    #loss.item() is a single value
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

end_time = datetime.now()
print(f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {str(end_time - start_time).split('.')[0]}")# %%