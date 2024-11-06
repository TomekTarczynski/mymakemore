import torch
import torch.nn.functional as F

############
# SETTINGS #
############

input_file = "names.txt"
learning_rate = 50
n_epochs = 200
n_batches = 1
n_inference = 50

##############
# READ INPUT #
##############

with open(input_file, 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]

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

data = []
for name in lines:
    data.append([0] + [char_into_int_map[c] for c in name] + [0])

xs = []
ys = []

for obs in data:
    for i in range(len(obs)-1):
        xs.append(obs[i])
        ys.append(obs[i+1])

##################
# NEURAL NETWORK #
##################

# Convert data to tensors
xs = torch.tensor(xs)
ys = torch.tensor(ys)
n_obs = xs.shape[0]
print(n_obs)

# Initialise
W = torch.randn(n_char, n_char, requires_grad=True)
input_x = F.one_hot(xs, num_classes = n_char).float()
batch_size = n_obs / n_batches

for i in range(n_epochs):
    for j in range(n_batches):
        batch_start = int(j * batch_size)
        batch_end = int((j+1) * batch_size)

        # Forward
        logits = input_x[batch_start:batch_end] @ W
        output = logits.exp()
        probs = output / output.sum(dim = 1, keepdim = True)
        pred = probs[torch.arange(batch_end - batch_start), ys[batch_start:batch_end]]
        loss = -pred.log().mean()
        print(f"Iter: {i} Batch: {j} Loss={loss:.4f}")

        # Backward
        W.grad = None
        loss.backward()
        W.data += -learning_rate * W.grad

# Final quality of the model
logits = input_x @ W
output = logits.exp()
probs = output / output.sum(dim = 1, keepdim = True)
pred = probs[torch.arange(n_obs), ys]
loss = -pred.log().mean()
print(f"\nFinal Loss={loss:.4f}")

# Model inference
print("\nModel inference:")
for i in range(n_inference):
    current_char = 0
    name = ''
    while True:
        input_x = F.one_hot(torch.tensor([current_char]), num_classes = n_char).float()
        logits =  input_x @ W
        output = logits.exp()
        probs = output / output.sum(dim = 1, keepdim = True)
        current_char = torch.multinomial(probs, num_samples=1, replacement=True).item()
        name += int_into_char_map[current_char]
        if current_char == 0:
            print(name)
            break

