# Importing
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Handling the data
df = pd.read_csv('games.csv')
df = df.drop(labels=['id', "created_at","last_move_at","turns","victory_status", "increment_code", "white_id", "black_id","moves","opening_eco","opening_name","opening_ply"], axis=1)
df = df[df['rated'] == True]
df.reset_index(drop=True, inplace=True)
df = df.drop(labels=["rated"], axis=1)

# The function to convert the winner into numbers
def returnThing(thing):
    return 0.0 if thing=="black" else 1.0
X = []
y = []
# Making x and y
for i in range(5, len(df)):
    X.append([
        float(df.iloc[i]['white_rating']),
        float(df.iloc[i]['black_rating'])
    ])
    y.append(returnThing(df.iloc[i]["winner"]))
# turning them into tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
# splitting them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Making the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 7 hidden layers
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        out = pytorch.sigmoid(self.out(x))

# Create the citerian, model, and optimizer
torch.manual_seed(41)
model = Model()
criterion = nn.BCELoss() # Standard for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train it
epochs = 1000

for i in range(epochs):
    # get result
    y_pred = model(X_train)
    # get loss
    loss = criterion(y_pred, y_train)
    # reset gradient
    optimizer.zero_grad()
    # go backwards and fix everything
    loss.backward()
    optimizer.step()
    # print it out every 10 epochs
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.item()}")
# get the test results
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predictions = (test_outputs >= 0.5).float()
    accuracy = (predictions == y_test).sum() / y_test.shape[0]
    print(f"Test Accuracy: {accuracy.item():.4f}")


# get the accuracy, and then we might be able to finally give it some data
with torch.no_grad():
    test_outputs = model([float(input("White rating")), float(input("Black rating"))])
    prediction = (test_outputs >= 0.5).float()
    print(prediction)
