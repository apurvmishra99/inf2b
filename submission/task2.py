import numpy as np
import scipy.io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import cm


def step(y):
    return 1 * (y > 0)


def task2_hNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    X = X.T
    bias = W[0]
    W = np.delete(W, 0, 0)
    # W.T shape == (1 x D)
    # X.shape == (D x N)
    Y = (W.T @ X) + bias
    # Y shape == (1 x N)
    return step(Y.T)

def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def task2_sNeuron(W, X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  W : (D+1)-by-1 vector of weights (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    X = X.T
    bias = W[0]
    W = np.delete(W, 0, 0)
    # W.T shape == (1 x D)
    # X shape == (D x N)
    Y = (W.T @ X) + bias
    # Y shape == (1 x N)
    return sigmoid(Y.T)

def task2_hNN_A(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # First layer weights
    W1, W2, W3, W4 = np.asanyarray([
                            [-2.0096131 ,  0.33077714,  1.        ],
                            [ 1.        , -0.45212818,  0.06827846],
                            [ 1.        , -0.18659937, -0.27743793],
                            [-1.20418437,  1.        , -0.40741555]])
       
    # Applying weights and calculating the outputs
    Y1 = task2_hNeuron(W1.reshape((3,1)), X)
    Y2 = task2_hNeuron(W2.reshape((3,1)), X)
    Y3 = task2_hNeuron(W3.reshape((3,1)), X)
    Y4 = task2_hNeuron(W4.reshape((3,1)), X)
    
    #  Second layer weight
    W5 = np.array([-3.5, 1, 1, 1, 1]).reshape((5,1))

    X_out = np.hstack((Y1,Y2,Y3,Y4))

    Y = task2_hNeuron(W5, X_out)

    return Y

    
def hNN(X, flag):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  flag : string denoting if the matrix corresponds to polygon A or B
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    if flag == "A":
        # First layer weights
        W1, W2, W3, W4 = np.asanyarray([
                                [-2.0096131 ,  0.33077714,  1.        ],
                                [ 1.        , -0.45212818,  0.06827846],
                                [ 1.        , -0.18659937, -0.27743793],
                                [-1.20418437,  1.        , -0.40741555]])
    else:
        # First layer weights
        W1, W2, W3, W4 = np.asanyarray([
                                [ 1.        ,  0.98977494, -0.53907994],
                                [ 1.        , -0.27213978, -0.03654737],
                                [ 1.        , -0.18637246, -0.17595583],
                                [-0.78936275,  0.2933849 ,  1.        ]])
              
    # Applying weights and calculating the outputs
    Y1 = task2_hNeuron(W1.reshape((3,1)), X)
    Y2 = task2_hNeuron(W2.reshape((3,1)), X)
    Y3 = task2_hNeuron(W3.reshape((3,1)), X)
    Y4 = task2_hNeuron(W4.reshape((3,1)), X)
    
    #  Second layer weight
    W5 = np.array([-3.5, 1, 1, 1, 1]).reshape((5,1))

    X_out = np.hstack((Y1,Y2,Y3,Y4))

    Y = task2_hNeuron(W5, X_out)

    return Y

def task2_hNN_AB(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Output from our network for points in A
    Y_A = hNN(X, "A")
    # Output from our network for points in B
    Y_B = hNN(X, "B")

    Y_A_B = np.hstack((Y_A,Y_B))

    # we are modelling a neuron with binary inputs x1 and x2 which
    # outputs 1 if and only if  x1 = 0 and x2 = 1, as seen in lectures
    # the line -x + y = 0 will be sufficient 
    # therefore,  w = [0, -1, 1]
    
    Y = task2_hNeuron(np.array([0,-1,1]).reshape((3,1)), Y_A_B)
    
    return Y

def sNN(X, flag):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    #  flag : string denoting if the matrix corresponds to polygon A or Bstring denoting if the matrix corresponds to polygon A or B
    # Output:
    #  Y : N-by-1 vector of output (np.double)
    
    if flag == "A":
        # First layer weights
        multiplication_factor = 350
        W1, W2, W3, W4 = multiplication_factor *  np.asanyarray([
                                [-2.0096131 ,  0.33077714,  1.        ],
                                [ 1.        , -0.45212818,  0.06827846],
                                [ 1.        , -0.18659937, -0.27743793],
                                [-1.20418437,  1.        , -0.40741555]])
        
    else:
        # First layer weights
        multiplication_factor = 300
        W1, W2, W3, W4 = multiplication_factor * np.asanyarray([
                                [ 1.        ,  0.98977494, -0.53907994],
                                [ 1.        , -0.27213978, -0.03654737],
                                [ 1.        , -0.18637246, -0.17595583],
                                [-0.78936275,  0.2933849 ,  1.        ]])
        
                      
    # Applying weights and calculating the outputs
    Y1 = task2_sNeuron(W1.reshape((3,1)), X)
    Y2 = task2_sNeuron(W2.reshape((3,1)), X)
    Y3 = task2_sNeuron(W3.reshape((3,1)), X)
    Y4 = task2_sNeuron(W4.reshape((3,1)), X)
    
    #  Second layer weight
    W5 = np.array([-3.5, 1, 1, 1, 1]).reshape((5,1))

    X_out = np.hstack((Y1,Y2,Y3,Y4))

    Y = task2_sNeuron(multiplication_factor *  W5, X_out)

    # For all outputs greater than 0.5 we classify them class 1
    return Y > 0.5

def task2_sNN_AB(X):
    # Input:
    #  X : N-by-D matrix of input vectors (in row-wise) (np.double)
    # Output:
    #  Y : N-by-1 vector of output (np.double)

    # Output from our network for points in A
    Y_A = sNN(X, "A")
    # Output from our network for points in B
    Y_B = sNN(X, "B")

    Y_A_B = np.hstack((Y_A,Y_B))

    # we are modelling a neuron with binary inputs x1 and x2 which
    # outputs 1 if and only if  x1 = 0 and x2 = 1, as seen in lectures
    # the line -x + y = 0 will be sufficient 
    # therefore,  w = [0, -1, 1]
    
    Y = task2_sNeuron(300 * np.array([0,-1,1]).reshape((3,1)), Y_A_B)
    
    return Y > 0.5

if __name__ == "__main__":

    # Task 2.5
    # Generating points between 0 and 7 to classify:
    xs = np.linspace(0, 7, 1000)
    ys = np.linspace(0, 7, 1000)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack((xx.ravel(), yy.ravel()))
    
    
    # Calling hNN_A on the generated points
    data = task2_hNN_A(grid.T)
    
    # Reshape the result to fit the plt function
    data = data.reshape((len(xs), len(ys)))
    
    # Setup the plot title, label and axis
    plt.title('Task 2.5 Plot')
    plt.xticks(np.arange(0, 10, 0.5), fontsize=7)
    plt.yticks(np.arange(0, 10, 0.5), fontsize=7)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    
    # Setup the legend
    blue_patch_legend = mpatches.Patch(color='lightblue', label='~A')
    purple_patch_legend = mpatches.Patch(color='purple', label='A')
    plt.legend(loc='best', fancybox=True, framealpha=0.1, handles=[
               blue_patch_legend, purple_patch_legend], facecolor='black', fontsize=12)
    
    # Plot, save as pdf and show result
    plt.contourf(xx, yy, data, cmap=cm.BuPu)
    plt.savefig('t2_regions_hNN_A.pdf')
    plt.show()

    # Task 2.7
    # Generating points between -2 and 7 to classify:
    xs = np.linspace(-2, 7, 1000)
    ys = np.linspace(-2, 7, 1000)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack((xx.ravel(), yy.ravel()))
    
    
    # Calling task2_hNN_AB on the generated points
    data = task2_hNN_AB(grid.T)
    
    # Reshape the result to fit the plt function
    data = data.reshape((len(xs), len(ys)))
    
    # Setup the plot title, label and axis
    plt.title('Task 2.7 Plot')
    plt.xticks(np.arange(-2, 10, 0.5), fontsize=7)
    plt.yticks(np.arange(-2, 10, 0.5), fontsize=7)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    
    # Setup the legend
    blue_patch_legend = mpatches.Patch(color='lightblue', label='Class 0')
    purple_patch_legend = mpatches.Patch(color='purple', label='Class 1')
    plt.legend(loc='best', fancybox=True, framealpha=0.1, handles=[
               blue_patch_legend, purple_patch_legend], facecolor='black', fontsize=12)
    
    # Plot, save as pdf and show result
    plt.contourf(xx, yy, data, cmap=cm.BuPu)
    plt.savefig('t2_regions_hNN_AB.pdf')
    plt.show()

    # Task 2.9
    # Generating points between -2 and 7 to classify:
    xs = np.linspace(-2, 7, 1000)
    ys = np.linspace(-2, 7, 1000)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack((xx.ravel(), yy.ravel()))
    
    
    # Calling task2_sNN_AB on the generated points
    data = task2_sNN_AB(grid.T)
    
    # Reshape the result to fit the plt function
    data = data.reshape((len(xs), len(ys)))
    
    # Setup the plot title, label and axis
    plt.title('Task 2.9 Plot')
    plt.xticks(np.arange(-2, 10, 0.5), fontsize=7)
    plt.yticks(np.arange(-2, 10, 0.5), fontsize=7)
    plt.xlabel('x1', fontsize=18)
    plt.ylabel('x2', fontsize=18)
    
    # Setup the legend
    blue_patch_legend = mpatches.Patch(color='lightblue', label='Class 0')
    purple_patch_legend = mpatches.Patch(color='purple', label='Class 1')
    plt.legend(loc='best', fancybox=True, framealpha=0.1, handles=[
               blue_patch_legend, purple_patch_legend], facecolor='black', fontsize=12)
    
    # Plot, save as pdf and show result
    plt.contourf(xx, yy, data, cmap=cm.BuPu)
    plt.savefig('t2_regions_sNN_AB.pdf')
    plt.show()
    