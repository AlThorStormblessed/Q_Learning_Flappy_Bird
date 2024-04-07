import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl, pandas as pd

with open("qtable-1708013638.pickle", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df = df.transpose()
# df = df[df.sum(axis=1, numeric_only=True) != 0]
df.to_csv(r'file.csv')