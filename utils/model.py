import numpy as np
import os
import joblib
import logging

class Perceptron:
  """Perceptron class
  """
  def __init__(self, eta: float=None, epochs: int=None):
    self.weights = np.random.rand(3)*1e-4 ## initialising random weights before training
    training = (eta is not None) and (epochs is not None)
    if training:
      logging.info(f"Initial weights before Training:\n{self.weights}")
    self.eta = eta
    self.epochs = epochs

  def _ZOutcome_(self,X,w):
    """Gives Dot product of input & weights

    Args:
        X (matrix): input data Feature matrix
        w (Vector):weights vector 

    Returns:
        Vector: Dot product of input & weights
    """
    return np.dot(X,w)

  def activation(self,z):
    """Applies the activation function
    """
    return np.where(z>0,1,0)

  def fit(self,X,y):
    """Fits Model on given input

    Args:
        X (pd.DataFrame): Input Independent features DataFrame
        y (pd.Series): Labe Feature
    """
    self.X =X
    self.y = y

  # To add bias to X
    X_with_bias = np.c_[self.X,-np.ones((len(self.X),1))]
    
    for epoch in range(self.epochs):
      logging.info(f"for epoch>> {epoch+1}")
      z = self._ZOutcome_(X_with_bias,self.weights)
      y_hat = self.activation(z)
      logging.info(f"Predicted value after Forward Pass: {y_hat}")

      # Error Calculation
      self.error = self.y - y_hat
      logging.info(f'Error:\n{self.error}')

      # Weight update
      self.weights = self.weights + self.eta* np.dot(X_with_bias.T,self.error)
      logging.info(f"Updated weights after {epoch+1}/{self.epochs}: {self.weights}")

  def predictFun(self,X):
    """Predictions function

    Args:
        X (pd.DataFrame): Input Independent features DataFrame

    Returns:
        np.array: Predicted features values
    """
    X_with_bias = np.c_[X,-np.ones((len(X),1))]
    z = self._ZOutcome_(X_with_bias,self.weights)
    return self.activation(z)
  
  def _create_dir(self,filename, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir,filename)

  def SaveModel(self,filename,model_dir=None):
    """Save model in the specified directory with specified filename

    Args:
        filename (str): Name of the model to be saved as 
        model_dir (path(str), optional): Location to be saved at. Defaults to None.
    """
    if model_dir:
      model_file_path = self._create_dir(filename,model_dir)
      joblib.dump(self,model_file_path)
    else:
      model_file_path = self._create_dir(filename,'model')
      joblib.dump(self,model_file_path)
    logging.info(f'Model saved at {model_file_path}')
  
  def loadModel_(self, filepath):
    """Load Trained Model from specified filepath

    Args:
        filepath (path(str)): Location to be saved at

    Returns:
        Model instance: returns model instance
    """
    return joblib.load(filepath)