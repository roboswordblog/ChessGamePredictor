import pandas as pd

df = pd.read_csv('games.csv')
df = df.drop(labels=['id', "created_at","last_move_at","turns","victory_status", "increment_code", "white_id", "black_id","moves","opening_eco","opening_name","opening_ply"], axis=1)
df = df[df['rated'] == True]
df.reset_index(drop=True, inplace=True)
df = df.drop(labels=["rated"], axis=1)

print(df.head())
