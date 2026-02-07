import numpy as np
from scipy.optimize import minimize

def keep_trying(func):
    success = False
    while not success:
        try:
            res = func()
            success = True
        except Exception as e:
            print(f"Error occurred: {e}")
    return res

class PCA_:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, x):
        # centering x about the ORIGIN (by subtracting each feature in a sample by the feature mean)
        self.mean_ = np.mean(x, axis=0)
        x_centered = x - self.mean_
        self.n_samples = x.shape[0]

        # eigenvalues/vectors
        Evals, EVs = keep_trying(lambda: self.get_all_eig(x_centered))
        sorted_idx = np.argsort(Evals)[::-1]
        Evals = Evals[sorted_idx]
        EVs = EVs[:, sorted_idx]

        if self.n_components is None:
            self.n_components = x.shape[1]  
        total_Evals = Evals.copy()
        EVs = EVs[:, :self.n_components]
        Evals = Evals[:self.n_components]

        self.components_ = EVs.T
        self.explained_variance_ = Evals
        total_var = np.sum(total_Evals)
        self.explained_variance_ratio_ = Evals/total_var

    def transform(self, x):
        x_centered = x - self.mean_
        x_pca = np.dot(x_centered, self.components_.T)
        return x_pca

    def inverse_transform(self, x_pca):
        x = np.dot(x_pca, self.components_) + self.mean_
        return x

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def get_eig(self, prev_EVs, samples):
        def objective(EV):
            Z = 0
            for data_pt in samples:
                Z += (EV@data_pt)**2
            return -Z
        def unit_norm(EV):
            return np.linalg.norm(EV) - 1
        
        constraints = [{'type': 'eq', 'fun': unit_norm}]
        for vec in prev_EVs:
            constraints.append({
                'type': 'eq',
                'fun': lambda EV, vec=vec: np.dot(EV, vec)
            })
        EV = np.random.randn(len(samples[0]))
        EV /= np.linalg.norm(EV)
        bounds = [(-1, 1)] * len(EV)
        res = minimize(objective, EV, constraints=constraints, bounds=bounds, options={'maxiter': int(1e9)})
        if not res.success:
            raise ValueError("Optimization error:", res.message)
        EV = res.x
        Eval = -res.fun/(len(samples)-1)
        return Eval, EV

    def get_all_eig(self, samples):
        n_features = len(samples[0])
        eig_vals = []
        eig_vecs = []
        prev_EVs = []

        for _ in range(n_features):
            Eval, EV = self.get_eig(prev_EVs, samples)
            eig_vals.append(Eval)
            eig_vecs.append(EV)
            prev_EVs.append(EV.copy())
        return np.array(eig_vals), np.array(eig_vecs)
    
