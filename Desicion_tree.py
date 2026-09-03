#%% 
## Classification Tree Code
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

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
        epsilon = 1e-10
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            b = 1
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

        return N * sum_total
    
    
    def custom_2(self, y_oh, left_indices, right_indices):
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
            
            b = np.sqrt(p_l)
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

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
        epsilon = 1e-10
        
        for l in range(num_classes):
            p_1l = left[:, l].sum() / N
            p_2l = right[:, l].sum() / N
            p_l = p_1l + p_2l
            
            b = p_l
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2


        return N * sum_total
    
    
    def custom_5(self, y_oh, left_indices, right_indices):
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
            
            b = p_l**2
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

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
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

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
            b = -(p_l**0.5) * np.log(max(p_l, epsilon))
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

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
            b = (-p_l)*np.log(max(p_l, epsilon))
            
            denominator_1 = max(p_1 * b**2, epsilon)
            denominator_2 = max(p_2 * b**2, epsilon)
            
            sum_total += ((p_1l - p_1 * p_l)**2) / denominator_1
            sum_total += ((p_2l - p_2 * p_l)**2) / denominator_2

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
                        'gain': gain
                    }                                                              
        
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

        return {'feature_index': best_split['feature_index'],
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
# synthetic datsets
def compare_metrics_generated_datasets(N, V, k, alpha, nmin, max_depth,
                                       n_datasets=50,test_size=0.25,
                                       sig_range=(0.05, 0.10),n_jobs=-1):

    criteria = ["entropy_sklearn","custom_1","custom_2","custom_3","custom_4","custom_5","custom_6","custom_7","custom_8"]

    columns = ["entropy_sklearn",
               "b = 1",
               "b = p_l ^ 0.5",
               "b = (p_l*(1 - p_l)) ^ 0.5",
               "b = p_l",
               "b = p_l ^ 2",
               "b = -log(p_l)",
               "b = -p_l^0.5 * log(p_l)",
               "b = -p_l * log(p_l)"]

    def run_dataset(seed):
        Nk, R, y, X, cen = generdat(N=N,V=V,k=k,alpha=alpha,nmin=nmin,seed=seed,sig_range=sig_range)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)
        y_oh_train = np.eye(k, dtype=float)[y_train]

        dataset_results = []

        for criterion in criteria:
            if criterion == "entropy_sklearn":
                model = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=2,min_samples_leaf=1,criterion="entropy",random_state=seed)
                model.fit(X_train, y_train)
            else:
                model = DecisionTree(max_depth=max_depth,min_samples_split=2,min_samples_leaf=1,criterion=criterion)
                model.fit(X_train, y_train, y_oh_train)

            y_pred = model.predict(X_test)
            scores = [accuracy_score(y_test, y_pred),precision_score(y_test,y_pred,average="weighted",zero_division=0),
                      recall_score(y_test,y_pred,average="weighted",zero_division=0),f1_score(y_test,y_pred,average="weighted",zero_division=0),
                      adjusted_rand_score(y_test, y_pred)]
            
            dataset_results.append(scores)

        return np.asarray(dataset_results).T

    all_results = Parallel(n_jobs=n_jobs)(delayed(run_dataset)(seed)for seed in range(1, n_datasets + 1))
    all_results = np.asarray(all_results)
    mean_results = np.round(all_results.mean(axis=0), 4)
    std_results = np.round(all_results.std(axis=0), 4)

    metrics = ["Accuracy","Precision","Recall","F1 score","ARI"]
    index = []
    table_data = []

    for metric_index, metric in enumerate(metrics):
        index.append((metric, "Mean"))
        table_data.append(mean_results[metric_index])
        index.append((metric, "Std"))
        table_data.append(std_results[metric_index])

    index = pd.MultiIndex.from_tuples(index,names=["Metric", "Statistic"])
    final_table = pd.DataFrame(table_data,columns=columns,index=index)    
    print("\n"f"N, V, k, alpha, nmin, max_depth, n_datasets = "f"{N, V, k, alpha, nmin, max_depth, n_datasets}")

    return final_table
#%%

## Mean/std of 50 exps.
# Real World datasets exp.

def compare_metrics_train_test(max_depth, X, y,*, N=None, V=None, k=None, alpha=None, nmin=None, n_jobs=-1):

    def run_seed(seed):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        encoder = OneHotEncoder(sparse_output=False)
        y_oh_train = encoder.fit_transform(y_train.reshape(-1, 1))

        custom_1 = DecisionTree(max_depth=max_depth, criterion='custom_1')
        custom_1.fit(X_train, y_train, y_oh_train)
        y_pred = custom_1.predict(X_test)
        accuracy_1, precision_1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_1, f1_1 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_1 = adjusted_rand_score(y_test, y_pred)

        sk_entropy = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=42)
        sk_entropy.fit(X_train, y_train)
        y_pred = sk_entropy.predict(X_test)
        accuracy_entropy_sk, precision_entropy_sk = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_entropy_sk, f1_entropy_sk = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_entropy_sk = adjusted_rand_score(y_test, y_pred)

        custom_2 = DecisionTree(max_depth=max_depth, criterion='custom_2')
        custom_2.fit(X_train, y_train, y_oh_train)
        y_pred = custom_2.predict(X_test)
        accuracy_2, precision_2 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_2, f1_2 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_2 = adjusted_rand_score(y_test, y_pred)

        custom_3 = DecisionTree(max_depth=max_depth, criterion='custom_3')
        custom_3.fit(X_train, y_train, y_oh_train)
        y_pred = custom_3.predict(X_test)
        accuracy_3, precision_3 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_3, f1_3 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_3 = adjusted_rand_score(y_test, y_pred)

        custom_4 = DecisionTree(max_depth=max_depth, criterion='custom_4')
        custom_4.fit(X_train, y_train, y_oh_train)
        y_pred = custom_4.predict(X_test)
        accuracy_4, precision_4 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_4, f1_4 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_4 = adjusted_rand_score(y_test, y_pred)

        custom_5 = DecisionTree(max_depth=max_depth, criterion='custom_5')
        custom_5.fit(X_train, y_train, y_oh_train)
        y_pred = custom_5.predict(X_test)
        accuracy_5, precision_5 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_5, f1_5 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_5 = adjusted_rand_score(y_test, y_pred)

        custom_6 = DecisionTree(max_depth=max_depth, criterion='custom_6')
        custom_6.fit(X_train, y_train, y_oh_train)
        y_pred = custom_6.predict(X_test)
        accuracy_6, precision_6 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_6, f1_6 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_6 = adjusted_rand_score(y_test, y_pred)

        custom_7 = DecisionTree(max_depth=max_depth, criterion='custom_7')
        custom_7.fit(X_train, y_train, y_oh_train)
        y_pred = custom_7.predict(X_test)
        accuracy_7, precision_7 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_7, f1_7 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_7 = adjusted_rand_score(y_test, y_pred)

        custom_8 = DecisionTree(max_depth=max_depth, criterion='custom_8')
        custom_8.fit(X_train, y_train, y_oh_train)
        y_pred = custom_8.predict(X_test)
        accuracy_8, precision_8 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_8, f1_8 = recall_score(y_test, y_pred, average='weighted', zero_division=0), f1_score(y_test, y_pred, average='weighted')
        ari_8 = adjusted_rand_score(y_test, y_pred)

        return np.round([[accuracy_entropy_sk, accuracy_1, accuracy_2,accuracy_3, accuracy_4, accuracy_5,accuracy_6, accuracy_7, accuracy_8],
                         [precision_entropy_sk, precision_1, precision_2,precision_3, precision_4, precision_5,precision_6, precision_7, precision_8],
                         [recall_entropy_sk, recall_1, recall_2,recall_3, recall_4, recall_5,recall_6, recall_7, recall_8],
                         [f1_entropy_sk, f1_1, f1_2,f1_3, f1_4, f1_5,f1_6, f1_7, f1_8],
                         [ari_entropy_sk, ari_1, ari_2,ari_3, ari_4, ari_5,ari_6, ari_7, ari_8]], 4)

    all_results = Parallel(n_jobs=n_jobs)(delayed(run_seed)(seed) for seed in range(1, 51))

    print(f'\nN, V, k, alpha, nmin, max_depth = {N, V, k, alpha, nmin, max_depth}')

    all_results = np.array(all_results)
    mean_results = np.round(np.mean(all_results, axis=0), 4)
    std_results = np.round(np.std(all_results, axis=0), 4)

    columns = ['entropy_sklearn',
               'b = 1',
               'b = p_l ^ 0.5',
               'b = (p_l*(1 - p_l)) ^ 0.5',
               'b = p_l',
               'b = p_l ^ 2',
               'b = -log(p_l)',
               'b = -p_l^0.5 * log(p_l)',     
               'b = -p_l * log(p_l)']
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'ARI']
    index_tuples = []

    for metric in metrics:
        index_tuples.append((metric, 'Mean'))
        index_tuples.append((metric, 'Std'))

    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Metric', 'Statistic'])

    final_table_data = []

    for i in range(len(metrics)):
        final_table_data.append(mean_results[i])
        final_table_data.append(std_results[i])

    return pd.DataFrame(final_table_data,columns=columns,index=multi_index)
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

    cen = (alpha - 1) + 2 * (1 - alpha) * np.random.rand(k, V)
    X = np.zeros((N, V))
    y = np.zeros(N, dtype=int)
    R = []
    
    sig_min, sig_max = sig_range
    start_idx = 0
    
    for k0 in range(k):
        nk = Nk[k0]
        end_idx = start_idx + nk
        
        R.append(range(start_idx, end_idx))
        y[start_idx:end_idx] = k0 
        
        sig = sig_min + (sig_max - sig_min) * np.random.rand(V)
        X[start_idx:end_idx] = np.random.randn(nk, V) * sig + cen[k0, :]
        
        start_idx = end_idx

    return Nk, R, y, X, cen

#%%
#--------------------------------------------------------------------------------------
# Example

N, V, k, alpha, nmin = 500, 3, 4, 0.5, 50
compare_metrics_generated_datasets(N=N,V=V,k=k,alpha=alpha,nmin=nmin,max_depth=3)
                                        