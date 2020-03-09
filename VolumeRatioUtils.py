import numpy as np
import LinConGauss as lcg
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing

class VolumeRatio:
    def __init__(self, dataset='mnist', N_start=10, N_finish=100, N_by=10, n_cores='all', label1=0, label2=1, corrupt_fraction=0, fourier_features = None):
        assert dataset in ['mnist', 'fashion-mnist'], 'Dataset must be one of either \'mnist\' or \'fashion-mnist\''
        
        self.dataset = dataset
        if self.dataset == 'mnist':
            self.dim = 28*28
        elif self.dataset == 'fashion-mnist':
            self.dim = 28*28
        self.N_start = N_start
        self.N_finish = N_finish
        self.N_by = N_by
        self.N_range = np.arange(self.N_start, self.N_finish+self.N_by,self.N_by)
        if n_cores == 'all':
            self.n_cores = int(multiprocessing.cpu_count())
        else:
            self.n_cores = n_cores
        self.label1 = label1
        self.label2 = label2
        self.corrupt_fraction = corrupt_fraction
        self.fourier_features = fourier_features
        self.projection_matrices = []
        #self.save_fn = save_fn
        self.samples = None
        self.test_accuracies = None
        
    def sample(self, n_samples=1000, save_fn=None):
        def get_data(n, label1 = 0, label2 = 1):
            '''
            get training data and A matrix
            '''
            if self.dataset == 'mnist':
                dataset = datasets.MNIST('./mnist', train=True, download=True)
            elif self.dataset == 'fashion-mnist':
                dataset = datasets.FashionMNIST('./fashion-mnist', train=True, download=True)
                
            data= 2*(dataset.data.numpy()/255 - .5)
            labels= dataset.targets.numpy()

            data = data[(labels==label1) | (labels==label2)].reshape(-1, self.dim)
            labels = labels[(labels==label1) | (labels==label2)]
            labels[labels == label1] = -1
            labels[labels == label2] = 1
            labels = labels.reshape(-1,1)
            if self.corrupt_fraction > 0:
                corrupt_ix = np.random.choice(labels.shape[0], int(self.corrupt_fraction*labels.shape[0]), replace=False)
                corrupt_labels = np.random.choice([-1,1], int(self.corrupt_fraction*labels.shape[0]))
                labels[corrupt_ix] = corrupt_labels

            scaler = StandardScaler()
            X = data
            X = scaler.fit_transform(X)
            y = labels
            Xn, yn = X[:n], y[:n]
            if self.fourier_features is not None:
                W = np.random.normal(size=(Xn.shape[1], self.fourier_features))
                self.projection_matrices.append({'n': n, 'W': W})
                Xn = np.cos(Xn@W)
            A = Xn*yn
            return Xn, yn, A
                
        def get_w_init(Xn, yn, n_init=10, eps=1e-4):
            '''
            gets initial samples in the domain 
            '''
            from sklearn import svm
            clf = svm.SVC(kernel='linear', C=1000)
            clf.fit(Xn, yn.flatten())
            w_hat = clf.coef_.flatten()
            inits = []
            inits.append(w_hat)
            for k in range(n_init-1):
                w_next = w_hat + np.random.uniform(-eps, eps, w_hat.shape)
                if np.all(np.sign(np.dot(Xn,w_next)) == yn.flatten()):
                    inits.append(w_next)
                else:
                    while not np.all(np.sign(np.dot(Xn,w_next)) == yn.flatten()):
                        w_next = w_hat + np.random.uniform(-eps, eps, w_hat.shape)
                    inits.append(w_next)
            return np.array(inits).T

        def samp(n):
            print('sampling n = %s' % str(n))
            Xn, yn, A = get_data(n, label1=self.label1, label2=self.label2)
            X_init = get_w_init(Xn, yn)
            b = np.zeros(n).reshape(n,1)
            lincon = lcg.LinearConstraints(A=A, b=b)
            sampler = lcg.sampling.EllipticalSliceSampler(n_iterations=n_samples,
                                                      linear_constraints=lincon,
                                                      n_skip=9,
                                                      x_init=X_init)
            sampler.run()
            samples = sampler.loop_state.X.T
            return samples
        
        results = Parallel(n_jobs=self.n_cores)(delayed(samp)(n) for n in self.N_range)
        
        self.samples = np.array(results)
        if save_fn is not None:
            save_samples(save_fn)
        
    def save_samples(self, save_fn):
        np.save(save_fn, self.samples)

    def compute_test_accuracies(self, m=5000, save_fn=None):
        assert self.samples is not None, 'Need to generate samples before computing test accuracy.'
    
        def get_test_set():
            assert self.fourier_features is None, 'Havent implemented test set computation for Fourier features'
            if self.dataset == 'mnist':
                dataset = datasets.MNIST('./mnist', train=True, download=True)
            elif self.dataset == 'fashion-mnist':
                dataset = datasets.FashionMNIST('./fashion-mnist', train=True, download=True)
                
            data= 2*(dataset.data.numpy()/255 - .5)
            labels= dataset.targets.numpy()

            data = data[(labels==self.label1) | (labels==self.label2)].reshape(-1, self.dim)
            labels = labels[(labels==self.label1) | (labels==self.label2)]
            labels[labels == self.label1] = -1
            labels[labels == self.label2] = 1
            labels = labels.reshape(-1,1)

            scaler = StandardScaler()
            X = data
            X = scaler.fit_transform(X)
            y = labels
            Xt, yt = X[self.dim:(self.dim+m)], y[self.dim:(self.dim+m)]
#             if self.fourier_features is not None:
#                 W = np.random.normal(size=(Xn.shape[1], self.fourier_features))
#                 Xn = np.cos(Xn@W)
#             A = Xn*yn
            return Xt, yt

        Xt, yt = get_test_set()
        accuracies = []
        for n in range(len(self.N_range)):
            curr_acc = []
            curr_samps = self.samples[n].T
            y_hat = (Xt@curr_samps).T
            for yy in y_hat:
                curr_acc.append(np.mean(np.sign(yy).flatten() == yt.flatten()))
            accuracies.append(curr_acc)
            
        self.test_accuracies = np.array(accuracies)
        if save_fn is not None:
            save_test_accuracies(save_fn)
            
    def save_test_accuracies(self, save_fn):
        np.save(save_fn, self.test_accuracies)
            
    
    

    
        

