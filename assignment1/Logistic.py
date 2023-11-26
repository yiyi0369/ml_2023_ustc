import numpy as np
from tqdm import tqdm

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0.01, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        ################################################################################
        # TODO:                                                                        #
        # Implement the sigmoid function.
        ################################################################################
        
        return 1/(1+np.exp(-x))
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
    def fit(self, X, y, lr=1e-4, tol=1e-7, max_iter=1e5,X_test=None,y_test=None):
        """
        Fit the regression coefficients via gradient descent or other methods 
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape (n_samples,), target data.
        - lr: float, learning rate for gradient descent.
        - tol: float, tolerance to decide convergence of gradient descent.
        - max_iter: int, maximum number of iterations for gradient descent.
        Returns:
        - losses: list, a list of loss values at each iteration.        
        """
        # If fit_intercept is True, add an intercept column
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # List to store loss values at each iteration
        losses = []
        accuracy=[]
        accuracy_test=[]
        for i in tqdm(range(int(max_iter))):
            acc=0
            loss=0
            linear_output = np.dot(X, self.coef_)
            y_pred=self.sigmoid(linear_output)
            
            loss=np.sum(-np.multiply(y,linear_output)+np.log(np.exp(linear_output)+1))
            # COMPUTE GRADIENT
            grad=np.dot(X.T,y_pred-y)
            if self.penalty=="l2":
                grad=grad-self.gamma*self.coef_
            elif self.penalty=="l1":
                grad=grad-self.gamma*np.sign(self.coef_)
            
            norm=np.linalg.norm(grad,ord=2)
            self.coef_=self.coef_-grad*lr
            #print(loss,norm,grad.shape)
            #acc=np.sum(1-np.abs(y_pred-y))/y.shape[0]
            for j in range(X.shape[0]):
                if y_pred[j]>=0.5:
                    y_pred[j]=1
                else:
                    y_pred[j]=0
            #if i != int(max_iter)-1:
                #accuracy.append(0)
                #losses.append(0)
                #accuracy_test.append(0)
                #continue
            acc=np.sum(1-np.abs(y_pred-y))/y.shape[0]
            acc_test=0
            losses.append(loss)
            
                
            accuracy.append(acc)
            if X_test is not None:
                y_pred_test=self.predict(X_test)
                acc_test=np.sum(1-np.abs(y_pred_test-y_test))/y_test.shape[0]
                accuracy_test.append(acc_test)
        ################################################################################
        # TODO:                                                                        #
        # Implement gradient descent with optional regularization.
        # 1. Compute the gradient 
        # 2. Apply the update rule
        # 3. Check for convergence
        ################################################################################
        
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return losses,accuracy,accuracy_test

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        
        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef_)
        probs=np.zeros(X.shape[0])
        
        ################################################################################
        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.
        ################################################################################

        for i in range(X.shape[0]):
            key=self.sigmoid(linear_output[i])
            if key >= 0.5:
                probs[i]=1
            else:
                probs[i]=0
        return probs
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
def get_acc(y_pred,y):
    acc=0
    for i in range(len(y_pred)):
        if y_pred[i]==y[i]:
            acc+=1
    return acc/len(y_pred)
