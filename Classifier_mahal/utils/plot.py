import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator

#from utils.model import decision_model 

def plot_data(dataset, colors=['red', 'blue']):

    fig, ax = plt.subplots()

    for current_class, current_color, in zip(np.unique(dataset.labels), colors):

        indices = np.where(current_class == dataset.labels)
        class_samples = dataset.samples[indices]

        ax.scatter(class_samples[:, 0], class_samples[:, 1], color = current_color, label = "Class %s" % current_class)

    ax.set_ylabel("X1")
    ax.set_xlabel("X0")
    ax.set_title("2D Visualization of Data")
    ax.legend()

    fig.tight_layout()
    #plt.show()


def plot_pdf_on_mesh(means, covariances, priors, x_min, x_max, y_min, y_max, res):
    

    # Generate a grid of points
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)

    # Initialize arrays to store the probabilities for each distribution
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    
    # Calculate the probabilities for each distribution at each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z1[i, j] = probability_density(point, means[0], covariances[0], priors[0])
            Z2[i, j] = probability_density(point, means[1], covariances[1], priors[1])

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the PDF surfaces
    ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.5)
    ax.plot_surface(X, Y, Z2, cmap='magma', alpha=0.5)

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    ax.set_title('Probability Denstiy on Mesh')

    plt.show()

def plot_pdf(dataset, pdf_vals, colors=['darkred','darkblue']):

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')

    for current_class, current_color, in zip(np.unique(dataset.labels), colors):

        indices = np.where(current_class == dataset.labels)
        class_samples = dataset.samples[indices]
        Z = np.asarray(pdf_vals)[indices]
        #X, Y = np.meshgrid(class_samples[:, 0], class_samples[:, 1])
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       #linewidth=0, antialiased=True)
        X = class_samples[:,0]
        Y = class_samples[:,1]

        ax.scatter(X, Y, Z, s=35, c=current_color, label="Class %s" % current_class)

    ax.set_ylabel("X1")
    ax.set_xlabel("X0")
    ax.set_title("Probability Density function")
    ax.legend()

    fig.tight_layout()
    #plt.show()

def linear(x, A,c):
    return (A.T @ x + c).flatten()


def quadratic(x, A,B,c):
    return (x @ A @ x.T+B.T @ x +c).flatten()


def plot_boundary(data,N,w1,b, w2=None):

    num = N  # Number of data points
    X1 = data[:N]
    X2 = data[N:]
    
    fig, ax = plt.subplots()

    # Plot data points
    ax.scatter(X1[:, 0], X1[:, 1], c='darkred', alpha=0.5, label='Class 0')
    ax.scatter(X2[:, 0], X2[:, 1], c='darkblue', alpha=0.5, label='Class 1')

    x = np.linspace(-15, 15, 100)
    y = np.linspace(-15, 15, 100)
    X, Y = np.meshgrid(x, y)

    points = np.column_stack([X.flatten(), Y.flatten()])
    Z_list=[]
    for point in points:
        if w2 is None:
            Z= linear(point, w1,b) 
            Z_list.append(Z)
        else:
            Z = quadratic(point, w1, w2, b)
            Z_list.append(Z)
    Z_list = np.array(Z_list)
    
    ax.contour(X, Y, Z_list.reshape(X.shape), levels=1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision Boundary on 2D')
    ax.grid(True)
    plt.show()


def plot_posterior(X,Y,sigmoidal):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, sigmoidal, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Posterior Probability')
    ax.set_title('Sigmoidal Shape of the Posterior Probability with Priors')
    plt.show()


def probability_density(data, mu, covariance, prior):

        pdf_vals=[]
        n = len(mu)
        det_covariance = np.linalg.det(covariance)
        coefficient = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_covariance))
        exponent = -0.5 * np.dot(np.dot((data - mu).T, np.linalg.inv(covariance)), (data - mu))
        density= (coefficient * np.exp(exponent))*np.asarray(prior)
        return density


