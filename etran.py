import torch
import numpy as np


def to_torch(ndarray):
    from collections.abc import Sequence
    if ndarray is None: return None
    if isinstance(ndarray, Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    if torch.is_tensor(ndarray): return ndarray
    raise ValueError('fail convert')


def Energy_Score(logits, percent, tail):
    logits = to_torch(logits)
    energy_score = torch.logsumexp(logits, dim=-1).numpy()
    if tail=='bot':
        chs = list(np.argsort(energy_score)[0:int(percent*10)*len(energy_score)//1000]) # #
    else:
        chs = list(np.argsort(energy_score)[-int(percent*10)*len(energy_score)//1000:])
    energy_score = energy_score[chs].mean() 
    return energy_score


class LDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _cov(self,X, shrinkage=-1):
      emp_cov = np.cov(np.asarray(X).T, bias=1)
      if shrinkage < 0:
          return emp_cov
      n_features = emp_cov.shape[0]
      mu = np.trace(emp_cov) / n_features
      shrunk_cov = (1.0 - shrinkage) * emp_cov
      shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
      return shrunk_cov


    def softmax(slf,X, copy=True):
      if copy:
          X = np.copy(X)
      max_prob = np.max(X, axis=1).reshape((-1, 1))
      X -= max_prob
      np.exp(X, X)
      sum_prob = np.sum(X, axis=1).reshape((-1, 1))
      X /= sum_prob
      return X

    def iterative_A(self,A, max_iterations=3):
      '''
      calculate the largest eigenvalue of A
      '''
      x = A.sum(axis=1)
      #k = 3
      for _ in range(max_iterations):
          temp = np.dot(A, x)
          y = temp / np.linalg.norm(temp, 2)
          temp = np.dot(A, y)
          x = temp / np.linalg.norm(temp, 2)
      return np.dot(np.dot(x.T, A), y)
    
    def _solve_eigen2(self, X, y, shrinkage):

        U,S,Vt = np.linalg.svd(np.float32(X), full_matrices=False)


        # solve Ax = b for the best possible approximate solution in terms of least squares
        self.x_hat2 = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

        y_pred1=X@self.x_hat1
        y_pred2=X@self.x_hat2

        scores_c = -np.mean((y_pred2 - y)**2)
        return scores_c,
    def _solve_eigen(self, X, y, shrinkage):

        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
      
        # X_ = pairwise_kernels(X, metric='linear')
        X_=X
       
        means = np.zeros(shape=(len(classes), X_.shape[1]))
        np.add.at(means, y, X_)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X_.shape[1], X_.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X_[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = self.iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        St = self._cov(X_, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter
        
        evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw)@Sb)
        
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        self.idx=np.argsort(evals)[0:len(X)//2]
   
 
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )


    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N
        '''
        # X,y,y_reg=self.sample_based_on_classes(X,y,y_reg)
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new #[:, : self._max_components]
   
        
    def energy_score(self, logits):
        logits = to_torch(logits)
        return torch.logsumexp(logits, dim=-1).numpy()

    def predict_proba(self, X,y):

        logits = np.dot(X, self.coef_.T) + self.intercept_
        scores = self.softmax(logits)
        return scores 

    def sample_based_on_classes(self,X,y,y_reg):
        import random
        X_new=[]
        y_new=[]

        labels=np.unique(y)
        mean_labels=np.zeros(len(labels))
        for label in labels:
          idx=np.where(y==label)
          X_label=X[idx]
          y_label=y[idx]
          y_label_reg=y_reg[idx]
          mean_labels[label]=np.mean(X_label)

        for label in labels:
          idx=np.where(y==label)
          X_label=X[idx]
          y_label=y[idx]
          y_label_reg=y_reg[idx]
          mean_label=np.mean(X_label)
          dist=0
          for label_ in labels:
            if label==label_:
              continue
            dist+=np.linalg.norm(X_label-mean_labels[label_],axis=-1)
          idx=np.argsort(dist)[len(X_label)//3:2*len(X_label)//3]
          if label==0:
            X_new=X_label[idx]
            y_new=y_label[idx]
            y_new_reg=y_label_reg[idx]
          else:
            X_new=np.append(X_new,X_label[idx],axis=0)
            y_new=np.append(y_new,y_label[idx],axis=0)
            y_new_reg=np.append(y_new_reg,y_label_reg[idx],axis=0)
        idx=np.arange(len(X_new))
        random.shuffle(idx)
        return X_new[idx],y_new[idx],y_new_reg[idx]


def LDA_Score(X,y):
  n = len(y)
  num_classes = len(np.unique(y))


  prob = LDA().fit(X, y).predict_proba(X,y)  # p(y|x)
  n = len(y)
  lda_score = np.sum(prob[np.arange(n), y]) / n
 
  return lda_score



