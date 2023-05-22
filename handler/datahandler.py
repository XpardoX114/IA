from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch.utils.data
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class HeartData(Dataset):
    def __init__(self, file_path):
        raw_data = pd.read_csv(file_path)
        x = raw_data.values[:, :-1]
        y = raw_data.values[:, -1]
        y = y.reshape(len(y), 1)
        one_hot_encoder = OneHotEncoder()
        min_max_scaler = MinMaxScaler()
        one_hot_encoder.fit(y)
        y = one_hot_encoder.transform(y).toarray()
        self.x = min_max_scaler.fit_transform(x)
        self.y = y

    def __len__(self):
        # e.g. len(x)
        return len(self.x)

    def __getitem__(self, idx):
        # e.g. foo[12]
        # idx would be 12 then
        return self.x[idx], self.y[idx]


def get_data():
    heart_data = HeartData("/home/alex/Documentos/handler/heart.csv")
    training_data, test_data = torch.utils.data.random_split(heart_data,[int(len(heart_data) * 0.7), len(heart_data) - int(len(heart_data) * 0.7)],)
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
    return train_dataloader, test_dataloader


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13,13),
            nn.ReLU(),
            nn.Linear(13, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.to(torch.float32))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.to(torch.float32))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


getdata = get_data()
train_dataloader = getdata[0]
test_dataloader = getdata[1]

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")