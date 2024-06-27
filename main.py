from utils import *
from model import *
from genetic import *

# //---------------------------------HYPERPARAMETERS---------------------------------//
path_model = "MLP_100.pth"

batch_size = 64
test_batch_size = 64
epochs = 5
seed = 1
log_interval = 500
TRAIN = False
no_cuda = True
hidden = 100

lr = 0.01
momentum = 0.5

# //---------------------------------UTILS---------------------------------//
use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")
print(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)

#split into train and validation
train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=test_batch_size, shuffle=True, **kwargs)

model = MLP(hidden_size=hidden).to(device)

# //---------------------------------TRAINING---------------------------------//
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
args = {}
args["log_interval"] = log_interval

if TRAIN:
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader, criterion)
        torch.save(model.state_dict(), path_model)
else:
    model.load_state_dict(torch.load(path_model))
    

q_model = coppy.deepcopy(model)

# //---------------------------------QUANTIZATION---------------------------------//
# TEST UNIFORM QUANTIZATION

# result = []
# print("Normal model")
# loss, acc, ms = testQuant(q_model, device, (val_loader, test_loader), criterion, quant=False)
# result.append((9, loss, acc, ms))

# for i in range(8, 0, -1):
#     print(f"Quantized model {i}b")
#     loss, acc, ms = testQuant(q_model, device, (val_loader, test_loader), criterion, quant=True, num_bits=i)
#     result.append((i, loss, acc, ms))

# print(result)

#TEST COMBINATORY SEARCH OF LAYER SPACE
# numbers = range(1, 9)
# cases = []
# # Generating all possible vectors of length 3
# for i in numbers:
#     for j in numbers:
#         for k in numbers:
#             vector = [i, j, k]
#             cases.append(vector)

# result = []
# import csv
# #create a csv file to store the results
# with open('brutal.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Generation","Candidate", "Fitness", "Accuracy"])
#     for case in tqdm(cases):
#         #print(f"Quantized model {case}b")
#         loss, acc, ms = testQuant(q_model, device, (val_loader, test_loader), criterion, quant=True, num_bits=case)
#         writer.writerow([0,','.join(map(str, case)), ','.join(map(str, [loss, ms])), acc])
#         result.append((case, loss, acc, ms))
# print(result)

# //---------------------------------GENETIC OPTIMIZATION---------------------------------//

genetic_quant(q_model, device, val_loader, criterion, neuron_opt=True)






