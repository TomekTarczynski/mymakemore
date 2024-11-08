import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from itertools import product

############
# SETTINGS #
############

input_file = "names.txt"
pct_train = 0.8
pct_valid = 0.1

n_char_used = 3
n_emb = 10
n_hidden = 200
n_batch = 100

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
    return torch.tensor(xs), torch.tensor(ys)
       
x_train, y_train = preprocess_data(data_train)
x_valid, y_valid = preprocess_data(data_valid)
x_test, y_test = preprocess_data(data_test)
print(f"{len(x_train)=} {len(y_train)=}")
print(f"{len(x_valid)=} {len(y_valid)=}")
print(f"{len(x_test)=} {len(y_test)=}")

##################
# NEURAL NETWORK #
##################

class linear:
    def __init__(self, n_in, n_out, bias = True):
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.is_training = True

        self.W = torch.randn(n_in, n_out)
        if bias:
            self.b = torch.randn(n_out)

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

class tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.tanh(x)
    
    def parameters(self):
        return []



class nn_mlp:
    def __init__(self, n_char_used, n_emb, n_hidden, n_char = 27):
        self.n_char_used = n_char_used
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_char = n_char

        self.C = torch.randn(self.n_char, self.n_emb)
        self.W1 = torch.randn(self.n_char_used * self.n_emb, self.n_hidden)
        self.b1 = torch.randn(self.n_hidden)
        self.W2 = torch.randn(self.n_hidden, self.n_char)
        self.b2 = torch.randn(self.n_char)

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, x):
        emb = self.C[x]
        emb = emb.view(-1, self.n_char_used * self.n_emb)
        h = torch.tanh(emb @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits

    def backward(self, loss, lr):
        for p in self.parameters:
            p.grad = None
        loss.backward()
        for p in self.parameters:
            p.data += -lr * p.grad

    def train(self, x, y, n_epochs, n_batch, lr):
        for i in range(n_epochs):
            batch_idx = torch.randint(0, x.shape[0], (n_batch, ))
   
            # FORWARD
            logits = self.forward(x[batch_idx])
            loss = F.cross_entropy(logits, y[batch_idx])
            self.backward(loss, lr)

n_char_used=3
n_emb=2
n_hidden=100
n_epochs=25000
n_batch=100
lr=0.1

mlp = nn_mlp(n_char_used=n_char_used, n_emb=n_emb, n_hidden=n_hidden, n_char=n_char)
mlp.train(x=x_train, y=y_train, n_epochs=n_epochs, n_batch=n_batch, lr=lr)
logits = mlp.forward(x_train)
loss = F.cross_entropy(logits, y_train)
print(loss.item())


C = torch.randn(n_char, n_emb)
layers = [  linear(n_in=n_char_used*n_emb, n_out=n_hidden), 
            tanh(),
            linear(n_in=n_hidden, n_out=n_char)]

parameters = [C]
for l in layers:
    parameters += l.parameters()

for p in parameters:
    p.requires_grad_() 

for i in range(n_epochs):
    batch_idx = torch.randint(0, x_train.shape[0], (n_batch, ))

    # FORWARD
    x = x_train[batch_idx]
    x = C[x].view(-1, n_char_used * n_emb)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y_train[batch_idx])

    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -lr * p.grad    

x = x_train
x = C[x].view(-1, n_char_used * n_emb)
for layer in layers:
    x = layer(x)
loss = F.cross_entropy(x, y_train)
print(loss.item())

    


