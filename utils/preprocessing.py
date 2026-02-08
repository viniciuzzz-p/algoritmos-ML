import numpy as np

class Min_Max_Scaler:

    def __init__(self):

        self.min = None
        self.max = None
    
    def fit(self, X):

        self.max = np.max(X, axis= 0)
        self.min = np.min(X, axis=0)

    def transform(self, X):

        denominador = self.max - self.min

        #previne que o denominador seja zero
        denominador[denominador == 0] = 1
        X_scaled = (X - self.min)/denominador

        return X_scaled
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)