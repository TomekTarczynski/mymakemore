import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from itertools import product
import time

############
# SETTINGS #
############

input_file = "names.txt"
pct_train = 0.8
pct_valid = 0.1

n_char_used=8
n_emb=10
n_hidden=128
n_epochs=500
n_batch=5000
lr=0.1

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
device = "cuda"

##############
# READ INPUT #
##############

with open(input_file, 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]
#lines = list(set(lines))

random.seed(42)
random.shuffle(lines)
n_obs_total = len(lines)
n_obs_train = int(n_obs_total * pct_train)
n_obs_valid = int(n_obs_total * pct_valid)
n_obs_test = n_obs_total - n_obs_train - n_obs_valid
print(f"{n_obs_total=}")
print(f"{n_obs_train=}")
print(f"{n_obs_valid=}")
print(f"{n_obs_test=}")

data_train = lines[:n_obs_train]
data_valid = lines[n_obs_train:(n_obs_train+n_obs_valid)]
data_test = lines[(n_obs_train+n_obs_valid):]

print(f"{len(data_train)=}")
print(f"{len(data_valid)=}")
print(f"{len(data_test)=}")

###################
# PREPROCESS DATA #
###################

unique_characters = ['.'] + sorted(list(set(list(''.join(lines)))))

n_char = len(unique_characters)
char_into_int_map = {}
int_into_char_map = {}
for index, char in enumerate(unique_characters):
    char_into_int_map[char] = index
    int_into_char_map[index] = char

def preprocess_data(data):
    xs = []
    ys = []

    for i in range(len(data)):
        word = [char_into_int_map[c] for c in data[i]] + [0]
        context = [0] * n_char_used
        for c in word:
            xs.append(context)
            ys.append(c)
            context = context[1:] + [c]
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)
       
x_train, y_train = preprocess_data(data_train)
x_valid, y_valid = preprocess_data(data_valid)
x_test, y_test = preprocess_data(data_test)
print(f"{len(x_train)=} {len(y_train)=}")
print(f"{len(x_valid)=} {len(y_valid)=}")
print(f"{len(x_test)=} {len(y_test)=}")

##################
# NEURAL NETWORK #
##################

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.n_in = in_features
        self.n_out = out_features
        self.bias = bias
        self.is_training = True

        self.W = (torch.rand(in_features, out_features, device=device) * 2 - 1) / (in_features**0.5)
        if bias:
            self.b = torch.zeros(out_features, device=device)

    def __call__(self, x):
        out = x @ self.W
        if self.bias:
            out += self.b
        self.out = out
        return self.out

    def parameters(self):
        if self.bias:
            return [self.W, self.b]
        else:
            return [self.W]

class BatchNorm1d:
    def __init__(self,num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.is_training = True

        self.gamma = torch.ones((1, self.num_features), device=device)
        self.beta = torch.zeros((1, self.num_features), device=device)

        if self.track_running_stats:
            self.run_mean = torch.zeros((1, self.num_features), device=device)
            self.run_var = torch.ones((1, self.num_features), device=device)

    def __call__(self, x):
        if self.is_training:
            if x.ndim==2:
                dim=0
            elif x.ndim==3:
                dim=(0,1)
            batch_mean = torch.mean(input=x, dim=dim, keepdim=True)
            batch_var = torch.var(input=x, dim=dim, keepdim=True)
            self.out = self.gamma * (x - batch_mean) / torch.sqrt(batch_var + self.eps) + self.beta

            if self.track_running_stats:
                with torch.no_grad():
                    self.run_mean = self.run_mean * (1 - self.momentum) + self.momentum * batch_mean
                    self.run_var = self.run_var * (1 - self.momentum) + self.momentum * batch_var
        else:
            self.out = self.gamma * (x - self.run_mean) / (self.run_var + self.eps) **0.5 + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.W = (torch.rand(self.num_embeddings, self.embedding_dim, device=device) * 2 - 1)

    def __call__(self, x):
        self.out = self.W[x]
        return self.out

    def parameters(self):
        return [self.W]

class FlattenConsecutive:
    def __init__(self, n_flatten=2):
        self.n_flatten=n_flatten

    def __call__(self, x):
        self.out = x.view(x.shape[0], x.shape[1]//self.n_flatten, x.shape[2]*self.n_flatten)
        if self.out.shape[1]==1:
            self.out=self.out.squeeze()
        return self.out

    def parameters(self):
        return []

class Tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

        for p in self.parameters():
            p.requires_grad_() 

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        parameters = []
        for l in self.layers:
            parameters += l.parameters()
        return parameters

### MY LAYERS CLASS ###

model = Sequential(
            [   Embedding(num_embeddings=n_char, embedding_dim=n_emb),
                
                FlattenConsecutive(),
                Linear(in_features=n_emb*2, out_features=n_hidden), 
                BatchNorm1d(num_features=n_hidden, momentum=0.001),
                Tanh(),

                FlattenConsecutive(),
                Linear(in_features=n_hidden*2, out_features=n_hidden), 
                BatchNorm1d(num_features=n_hidden, momentum=0.001),
                Tanh(),

                FlattenConsecutive(),
                Linear(in_features=n_hidden*2, out_features=n_hidden), 
                BatchNorm1d(num_features=n_hidden, momentum=0.001),
                Tanh(),
             
                Linear(in_features=n_hidden, out_features=n_char),
                BatchNorm1d(num_features=n_char, momentum=0.001)])

start_time = time.time()
for i in range(n_epochs):
    batch_idx = torch.randint(0, x_train.shape[0], (n_batch, ), device=device)

    # FORWARD
    logits = model(x_train[batch_idx])
    loss = F.cross_entropy(logits, y_train[batch_idx])

    for p in model.parameters():
        p.grad = None
    loss.backward()
    for p in model.parameters():
        p.data += -lr * p.grad
end_time = time.time()        
print("Computation time:", end_time - start_time, "seconds")

logits_train = model(x_train)
loss_train = F.cross_entropy(logits_train, y_train)

logits_valid = model(x_valid)
loss_valid = F.cross_entropy(logits_valid, y_valid)
print(f"{loss_train.item()=} {loss_valid.item()=}")

for l in model.layers:
    print(f"{l.__class__} {l.out.shape}")