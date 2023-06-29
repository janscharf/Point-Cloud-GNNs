import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

#Turn input data of chip coordinates into real cartesian coordinates
#Transposing Dictionaries
def change_to_real_coordinates_for_one_event(layer, row, col):

    def lane2layer(lane):
        const={40:22,39:22,42:20,41:20,44:18,43:18,46:16,45:16,
            48:14,47:14,50:12,49:12,52:10,51:10,54: 8,53: 8,
            38: 6,55: 6,36: 4,37: 4,32: 0,35: 0,34: 2,33: 2,
            64:23,63:23,66:21,65:21,68:19,67:19,70:17,69:17,
            72:15,71:15,74:13,73:13,76:11,75:11,78: 9,77: 9,
            62: 7,79: 7,60: 5,61: 5,56: 1,59: 1,58: 3,57: 3}
        return const[lane]

    def lane2chipid(lane):
        const={
            40: 0,39: 1,42: 2,41: 3,44: 4,43: 5,46: 6,45: 7,
            48: 8,47: 9,50:10,49:11,52:12,51:13,54:14,53:15,
            38:16,55:17,36:18,37:19,32:20,35:21,34:22,33:23,
            64:24,63:25,66:26,65:27,68:28,67:29,70:30,69:31,
            72:32,71:33,74:34,73:35,76:36,75:37,78:38,77:39,
            62:40,79:41,60:42,61:43,56:44,59:45,58:46,57:47
        }
        return const[lane]

    def layer2isInv(layerNr):
        if layerNr%2==1:
            return True
        else:
            return False

    def IsLeftChip(lane):
        layerNr=lane2layer(lane)
        isInv = layer2isInv(layerNr)
        chipid= lane2chipid(lane)
        if chipid%2 ==1:
            isOdd=True
        else:
            isOdd=False
        if isOdd!=isInv:
            return True
        else:
            return False

    def checkforsides(lane, row, col):
        if not IsLeftChip(lane):
            col= 1023-col
            row= -row-1+512
        else:
            row=row+512
        layer= lane2layer(lane)
        return layer, row, col

    for i in range(layer.shape[0]):
        layer[i], row[i], col[i]=checkforsides(layer[i], row[i], col[i])
    return layer, row, col

def combine(layer, row, col):
    data=[]
    for i in tqdm(range(layer.shape[0])):
        rep=np.stack((row[i], col[i], layer[i]), axis=-1)
        rep=rep.tolist()
        data.append(rep)
    return data #shape(row,col,layer)->(x,y,z)

def plot_one_event(data, title, path):
    X=[]
    Y=[]
    Z=[]
    for k in tqdm(range(24)):
        for i in tqdm(range(1024)):
            for j in range(1024):
                if data[i, j, k]!=0:
                    X.append(i)
                    Y.append(j)
                    Z.append(k)

    figure = plt.figure(figsize = (15, 12))
    subplot3d = plt.subplot(111, projection='3d')
    subplot3d.axes.set_xlim3d(left=0, right=1024)
    subplot3d.axes.set_ylim3d(bottom=0, top=1024)
    subplot3d.axes.set_zlim3d(bottom=0, top=24)
    subplot3d.set_xlabel('rows', fontweight = 'bold')
    subplot3d.set_ylabel('columns', fontweight = 'bold')
    subplot3d.set_zlabel('layers', fontweight = 'bold')
    my_cmap = plt.get_cmap('hsv')
    scatter = subplot3d.scatter(X, Y, Z, c=Z, cmap=my_cmap)
    subplot3d.set_title(title)
    print("Hi", path)
    plt.savefig(path)
    print("Event "+ name + " was plotted and saved to " + path + " .")

def plot_one_event_from_graph_data(data, title, path):
    X=[]
    Y=[]
    Z=[]
    for k in tqdm(range(data.shape[0])):
                    X.append(data[k, 1])#rows
                    Y.append(data[k, 2])#col
                    Z.append(data[k, 0])

    figure = plt.figure(figsize = (15, 12))
    subplot3d = plt.subplot(111, projection='3d')
    subplot3d.axes.set_xlim3d(left=0, right=1024)
    subplot3d.axes.set_ylim3d(bottom=0, top=1024)
    subplot3d.axes.set_zlim3d(bottom=0, top=24)
    subplot3d.set_xlabel('rows', fontweight = 'bold')
    subplot3d.set_ylabel('columns', fontweight = 'bold')
    subplot3d.set_zlabel('layers', fontweight = 'bold')
    my_cmap = plt.get_cmap('hsv')
    scatter = subplot3d.scatter(X, Y, Z, c=Z, cmap=my_cmap)
    subplot3d.set_title(title)
    print("Hi", path)
    plt.savefig(path)
    print("Event "+ name + " was plotted and saved to " + path + " .")
    plt.close()

def plot_loss(loss_d_real, loss_d_noise, loss_g_noise):
    X = np.arange(len(loss_d_real))
    plt.plot(X, loss_d_real, "r--", label="discriminator real data")
    plt.plot(X, loss_d_noise, "b--", label="discriminator noise data")
    plt.plot(X, loss_g_noise, "g-", label="GAN noise data")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig("/u/jscharf/Documents/Daten/Generativ/Loss_plot.png")
    plt.close()
