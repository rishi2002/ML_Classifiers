import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice

"""Read the data to a pandas DataFrame"""
def read_data():
    data = pd.read_csv('data.csv',header=None)
    data.columns=['X1','X2']
    return data

"""Plot full unclustered data"""
def plot(data):
    plt.scatter(data['X1'],data['X2'], s=5)
    plt.show()

"""Plot clusters predicted by the algorithm"""
def plot_clustered(data, y_pred):
        
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )   

    colors = np.append(colors, ["#000000"])
    data = data.to_numpy()
    plt.scatter(data[:, 0], data[:, 1], s=5, color=colors[y_pred])
    plt.show()


"""Write clusters to .txt file"""
def save_clusters(data,name):
    with open(name, "w") as f:
            for i in data:
                f.write(str(i)+'\n')