import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
df = pd.read_csv('games.csv')
df = df.drop(labels=['id', "created_at","last_move_at","turns","victory_status", "increment_code", "white_id", "black_id","moves","opening_eco","opening_name","opening_ply"], axis=1)
df = df[df['rated'] == True]
df.reset_index(drop=True, inplace=True)
df = df.drop(labels=["rated"], axis=1)

X = []
y = []

def returnThing(thing):
    return 1.0 if thing=="black" else 0.0

for i in range(5, len(df)):
    X.append(float(df.iloc[i]['white_rating']))
    X.append(float(df.iloc[i]['black_rating']))
    y.append(returnThing(df.iloc[i]["winner"]))


X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = self.out(x)
