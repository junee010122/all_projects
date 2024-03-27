import numpy as np

from utils.data import load_datasets, make_class_data
from utils.plot import plot_boundary, plot_posterior


class decision_model():

    def __init__(self, params):
        super().__init__()

    import numpy as np

    def get_M_distance(x1, x2=None, Sig=None, mu=None):

        if x2 is None and Sig is None and mu is None:
            raise ValueError("Please provide either x2 and Sig or mu")

        if Sig is not None and mu is not None and x2 is None:
            diff = x1 - mu
        
            det = Sig[0][0] * Sig[1][1] - Sig[0][1] * Sig[1][0]
            Sig_inv = np.array([[Sig[1][1], -Sig[0][1]], [-Sig[1][0], Sig[0][0]]]) / det
            m_distance = np.sqrt(np.dot(np.dot(diff.T, Sig_inv), diff))
            return m_distance

        if Sig is not None and x2 is not None:
            diff_x1 = x1 - mu
            diff_x2 = x2 - mu

            adj = np.zeros_like(Sig, dtype=float)
            for i in range(len(Sig)):
                for j in range(len(Sig)):
                    temp = Sig.copy()
                    temp = np.delete(temp, i, axis=0)
                    temp = np.delete(temp, j, axis=1)
                    cofactor = (-1)**(i+j) * (temp[0, 0] * temp[1, 1] - temp[0, 1] * temp[1, 0])
                    adj[i, j] = cofactor
            det = Sig[0][0] * Sig[1][1] - Sig[0][1] * Sig[1][0]
            Sig_inv = adj / det
            m_distance = np.sqrt(np.dot(np.dot(diff_x1.T, Sig_inv), diff_x1) + np.dot(np.dot(diff_x2.T, Sig_inv), diff_x2))
            return m_distance
  
    def probability_density(data, mu, covariance, prior):

        pdf_vals=[]
        n = len(mu)
        det_covariance = np.linalg.det(covariance)
        coefficient = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_covariance))
        exponent = -0.5 * np.dot(np.dot((data - mu).T, np.linalg.inv(covariance)), (data - mu))
        density= (coefficient * np.exp(exponent))*np.asarray(prior)
        return density
    
    def Discriminant_function(m_dist,dim, cov, prior):
        discriminant_val =-0.5 * (m_dist ** 2) - 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)
        
        return discriminant_val

    def decision_boundary(data, mu, covariance, prior):

        covariance = np.array(covariance)
        mu = np.array(mu)

        if (mu[0]==mu[1]).all():
            w1 = np.dot(np.linalg.inv(covariance[0]), mu[0])
            b = -1/2 * (np.dot(mu[0].T, np.dot(np.linalg.inv(covariance[0]), mu[0]))) +  np.log(prior[0]/prior[1])

            plot_boundary(data=data,N=1000,w1=w1, w2=None, b=b)
        else:
            w1 = -1/2*(np.linalg.inv(covariance[0])-np.linalg.inv(covariance[1]))
            w2 = np.dot(np.linalg.inv(covariance[0]), mu[0]) - np.dot(np.linalg.inv(covariance[1]), mu[1])
            b = -1/2 * (np.dot(mu[0].T, np.dot(np.linalg.inv(covariance[0]), mu[0])) - np.dot(mu[1].T, np.dot(np.linalg.inv(covariance[1]), mu[1]))) + np.log(prior[0]/prior[1])-1/2*(np.log(np.linalg.det(covariance[0]))-np.log(np.linalg.det(covariance[1])))

            plot_boundary(data=data,N=1000, w1=w1,w2=w2,b=b)

    def posterior_prob(mean1, cov1, mean2, cov2, prior1, prior2, resolution=100):

        x = np.linspace(-10, 10, resolution)
        y = np.linspace(-10, 10, resolution)
        X, Y = np.meshgrid(x, y)

        det_cov1 = cov1[0][0] * cov1[1][1] - cov1[0][1] * cov1[1][0]
        det_cov2 = cov2[0][0] * cov2[1][1] - cov2[0][1] * cov2[1][0]

        posterior1 = (prior1 * np.exp(-0.5 * ((X - mean1[0])**2 + (Y - mean1[1])**2) / det_cov1)) / ((2 * np.pi) ** 2 * det_cov1 ** 0.5)
        posterior2 = (prior2 * np.exp(-0.5 * ((X - mean2[0])**2 + (Y - mean2[1])**2) / det_cov2)) / ((2 * np.pi) ** 2 * det_cov2 ** 0.5)

        sigmoidal = posterior1 / (posterior1 + posterior2)


        plot_posterior(X,Y,sigmoidal)









