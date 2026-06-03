#%% 
## Classification Tree Code
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None                                                          
        self.feature_importances = None                                          


    def entropy(self, y):
        counts = np.bincount(y)                                                  
        probabilities = counts / len(y)                                          
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])         


    def gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)


    def information_gain(self, y, left_indices, right_indices):
        if self.criterion == 'entropy':                                          
            impurity_func = self.entropy
        elif self.criterion == 'gini':
            impurity_func = self.gini
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        parent_impurity = impurity_func(y)                                       
        left_impurity = impurity_func(y[left_indices])
        right_impurity = impurity_func(y[right_indices])

        n, n_left, n_right = len(y), len(left_indices), len(right_indices)
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        inf_gain = parent_impurity - weighted_impurity
        
        return inf_gain                                                          
    
    
    def custom_1(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()

        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                               
        sum_total = 0
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = 1
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total
    
    
    def custom_2(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                             
        sum_total = 0
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = np.sqrt(p_l)
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total
    

    def custom_3(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                              

        sum_total = 0
        epsilon = 1e-10 
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = np.sqrt(p_l*(1 - p_l))
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

        return N * sum_total
    

    def custom_4(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                               
        sum_total = 0
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = p_l
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total
    
    
    def custom_5(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]
        sum_total = 0
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = p_l**2
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total
    

    def custom_6(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                              
        sum_total = 0
        epsilon = 1e-10
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = -np.log(max(p_l, epsilon))
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total
    

    def custom_7(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                               

        sum_total = 0
        epsilon = 1e-10
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l

            b = (-p_l)*np.log(max(p_l, epsilon))
            
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total
    
    
    def custom_8(self, y_oh, left_indices, right_indices):
        N = y_oh.sum()
        left = y_oh[left_indices]
        right = y_oh[right_indices]
        p_1 = left.sum() / N
        p_2 = right.sum() / N
        num_classes = y_oh.shape[1]                                               

        sum_total = 0
        epsilon = 1e-10
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            
            b = -(p_l**0.5) * np.log(max(p_l, epsilon))
            
            sum_total += ((p_1l - p_1 * p_l)**2) / p_1 * b**2
            sum_total += ((p_2l - p_2 * p_l)**2) / p_2 * b**2

        return N * sum_total    
    

    def most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]


    def find_best_split(self, X, y, num_features, y_oh=None):
        best_gain = -float('inf')                                                  
        best_split = None                                                          

        for feature_index in range(num_features):                                  
            feature_values = np.sort(X[:, feature_index])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2     
            
            for threshold in thresholds:                                          
                left_indices = np.where(X[:, feature_index] <= threshold)[0]      
                right_indices = np.where(X[:, feature_index] > threshold)[0]      

                if (len(left_indices) < self.min_samples_leaf or 
                    len(right_indices) < self.min_samples_leaf):
                    continue                                                      

                if self.criterion == 'custom_1':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_1 criterion")
                    gain = self.custom_1(y_oh, left_indices, right_indices)
                
                elif self.criterion == 'custom_2':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_2 criterion")
                    gain = self.custom_2(y_oh, left_indices, right_indices)
                
                elif self.criterion == 'custom_3':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_3 criterion")
                    gain = self.custom_3(y_oh, left_indices, right_indices)                    
                
                elif self.criterion == 'custom_4':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_4 criterion")
                    gain = self.custom_4(y_oh, left_indices, right_indices)
                
                elif self.criterion == 'custom_5':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_5 criterion")
                    gain = self.custom_5(y_oh, left_indices, right_indices)
                
                elif self.criterion == 'custom_6':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_6 criterion")
                    gain = self.custom_6(y_oh, left_indices, right_indices)    
                    
                elif self.criterion == 'custom_7':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_7 criterion")
                    gain = self.custom_7(y_oh, left_indices, right_indices)
                    
                elif self.criterion == 'custom_8':
                    if y_oh is None:
                        raise ValueError("y_oh required for custom_7 criterion")
                    gain = self.custom_8(y_oh, left_indices, right_indices)                      
                
                else:
                    gain = self.information_gain(y, left_indices, right_indices)  

                if gain > best_gain:                                               
                    best_gain = gain                                               
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'gain': gain}                                                              
        
        return best_split                                                          


    def fit(self, X, y, y_oh=None):
        num_features = X.shape[1]
        self.feature_importances = np.zeros(num_features)                          
        self.tree = self.grow_tree(X, y, y_oh, depth=0)
        total = self.feature_importances.sum()
        
        if total > 0:
            self.feature_importances /= total


    def grow_tree(self, X, y, y_oh, depth):
        num_samples, num_features = X.shape
        num_classes = len(set(y))

        if (depth == self.max_depth or 
            num_classes == 1 or 
            num_samples < self.min_samples_split):
            return self.most_common_label(y)

        if self.criterion.startswith('custom_'):
            best_split = self.find_best_split(X, y, num_features, y_oh)
        else:
            best_split = self.find_best_split(X, y, num_features)

        if best_split is None:
            return self.most_common_label(y)

        left_indices, right_indices = best_split['left_indices'], best_split['right_indices']
        
        if self.criterion == 'custom_1':
            gain = self.custom_1(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_2':
            gain = self.custom_2(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_3':
            gain = self.custom_3(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_4':
            gain = self.custom_4(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_5':
            gain = self.custom_5(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_6':
            gain = self.custom_6(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_7':
            gain = self.custom_7(y_oh, left_indices, right_indices)
        elif self.criterion == 'custom_8':
            gain = self.custom_8(y_oh, left_indices, right_indices)            
            
        else:
            gain = self.information_gain(y, left_indices, right_indices)

        self.feature_importances[best_split['feature_index']] += gain              

        left_subtree = self.grow_tree(X[left_indices], y[left_indices], 
                                    y_oh[left_indices] if y_oh is not None else None, 
                                    depth + 1)
        right_subtree = self.grow_tree(X[right_indices], y[right_indices], 
                                     y_oh[right_indices] if y_oh is not None else None, 
                                     depth + 1)

        return {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree}


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])


    def _traverse_tree(self, x, node):
        if isinstance(node, dict):
            if x[node['feature_index']] <= node['threshold']:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])

        return node     
    
#%%

## Mean/std of 50 exps.

def compare_metrics_train_test(max_depth, X, y, *, N=None, V=None, k=None, alpha=None, nmin=None, n_jobs=-1):

    def run_one_seed(seed):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

        encoder = OneHotEncoder(sparse_output=False)
        y_oh_train = encoder.fit_transform(y_train.reshape(-1, 1))

        sk_entropy = DecisionTreeClassifier(max_depth=max_depth,criterion='entropy',random_state=seed)
        sk_entropy.fit(X_train, y_train)
        y_pred = sk_entropy.predict(X_test)

        accuracy_entropy_sk = accuracy_score(y_test, y_pred)
        precision_entropy_sk = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_entropy_sk = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_entropy_sk = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        ari_entropy_sk = adjusted_rand_score(y_test, y_pred)

        custom_results = []

        for criterion in [
            'custom_1',
            'custom_2',
            'custom_3',
            'custom_4',
            'custom_5',
            'custom_6',
            'custom_7',
            'custom_8']:
            
            model = DecisionTree(max_depth=max_depth, criterion=criterion)
            model.fit(X_train, y_train, y_oh_train)
            y_pred = model.predict(X_test)

            custom_results.append([
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='weighted', zero_division=0),
                recall_score(y_test, y_pred, average='weighted', zero_division=0),
                f1_score(y_test, y_pred, average='weighted', zero_division=0),
                adjusted_rand_score(y_test, y_pred)])

        custom_results = np.array(custom_results).T
        results = np.column_stack([[accuracy_entropy_sk, precision_entropy_sk, recall_entropy_sk, f1_entropy_sk, ari_entropy_sk],
                                   custom_results])

        return np.round(results, 4)

    all_results = Parallel(n_jobs=n_jobs)(delayed(run_one_seed)(seed) for seed in range(1, 51))

    print(f'\nN, V, k, alpha, nmin, max_depth = {N, V, k, alpha, nmin, max_depth}')

    all_results = np.array(all_results)

    mean_results = np.round(np.mean(all_results, axis=0), 4)
    std_results = np.round(np.std(all_results, axis=0), 4)

    columns = ['entropy_sklearn','b = 1','b = p_v ^ 0.5','b = (p_v*(1 - p_v)) ^ 0.5','b = p_v','b = p_v ^ 2','b = -log(p_v)','b = -p_v * log(p_v)','b = -p_v^0.5 * log(p_v)']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'ARI']

    index_tuples = []
    for metric in metrics:
        index_tuples.append((metric, 'Mean'))
        index_tuples.append((metric, 'Std'))

    multi_index = pd.MultiIndex.from_tuples(index_tuples,names=['Metric', 'Statistic'])

    final_table_data = []
    for i in range(len(metrics)):
        final_table_data.append(mean_results[i])
        final_table_data.append(std_results[i])

    final_table = pd.DataFrame(final_table_data,columns=columns,index=multi_index)

    return final_table     

#%%

# Data generator

# Parameters:
# - N: Total number of data points
# - V: Number of dimensions/features
# - k: Number of clusters
# - alpha: Controls cluster center spread (centers are in [α-1, 1-α])
# - nmin: Minimum points per cluster
# - seed: Random seed for reproducibility
# - sig_range: Tuple (min, max) for cluster standard deviations

# Returns:
# - Nk: Array of cluster sizes
# - R: List of ranges for each cluster
# - y: Cluster labels for each point
# - X: Generated data (N x V array)
# - cen: Cluster centers (k x V array)        

def generdat(N, V, k, alpha, nmin, seed=None, sig_range=(0.05, 0.1)):
    if N < k * nmin:
        raise ValueError(f"N must be >= k * nmin. Got N={N}, k={k}, nmin={nmin}")
    if k < 1:
        raise ValueError("k must be at least 1")
    if alpha == 1:
        raise ValueError("alpha cannot be 1")

    if seed is not None:
        np.random.seed(seed)

    if k == 1:
        Nk = np.array([N])
    else:
        base_sizes = np.ones(k, dtype=int) * nmin
        remaining = N - k * nmin
        if remaining > 0:
            additional = np.random.multinomial(remaining, np.ones(k)/k)
            Nk = base_sizes + additional
        else:
            Nk = base_sizes

    # Cluster centers
    cen = (alpha - 1) + 2 * (1 - alpha) * np.random.rand(k, V)

    X = np.zeros((N, V))
    y = np.zeros(N, dtype=int)
    R = []
    
    sig_min, sig_max = sig_range
    start_idx = 0
    
    for k0 in range(k):
        nk = Nk[k0]
        end_idx = start_idx + nk
        
        # Range for the current cluster
        R.append(range(start_idx, end_idx))
        y[start_idx:end_idx] = k0 
        
        # Cluster data generation
        sig = sig_min + (sig_max - sig_min) * np.random.rand(V)
        X[start_idx:end_idx] = np.random.randn(nk, V) * sig + cen[k0, :]
        
        start_idx = end_idx

    return Nk, R, y, X, cen

#%%
#--------------------------------------------------------------------------------------
# Example

N, V, k, alpha, nmin = 500, 3, 4, 0.5, 50
Nk, R, y, X, cen = generdat(N, V, k, alpha, nmin, seed=101)

compare_metrics_train_test(max_depth=3, X=X, y=y, N=N, V=V, k=k, alpha=alpha, nmin=nmin)
                                        