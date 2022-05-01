import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import pandas as pd
from utils.model import Perceptron
import logging


#Logging Configuration
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir,'running_logs.log'),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
    )


def Data_prepare(df, target='y'):
    logging.info('Preparing Data to retrun X & y Features')
    X = df.drop(target,axis=1)
    y = df[target]

    return X,y

def save_plot(df, model, filename="plot.png", plot_dir="plots"):
    def _create_base_plot(df):
        logging.info('Creating Base Plot')
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        
        figure = plt.gcf()
        figure.set_size_inches(10, 8)
    
    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        logging.info("Plotting decision regions")
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)
        
        X = X.values # as an array
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        x1_min, x1_max = x1.min() - 1, x1.max() + 1 
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                              )
        y_hat = classifier.predictFun(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.plot()
        
    X, y = Data_prepare(df)
    
    _create_base_plot(df)
    _plot_decision_regions(X, y, model)
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    logging.info(f"Saving Plot at {plot_path}")

#defining main function
def main(data, modelName,plotName,eta,epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is Raw Dataset: \n{df}")
    X,y = Data_prepare(df)

    # Model initialising & Fittign
    model = Perceptron(eta=eta,epochs=epochs)
    model.fit(X,y)

    # Save,Load& Predict from model
    model.SaveModel(filename=modelName)
    # loaded_model_AND = Perceptron().loadModel_(filepath='/content/model/model_AND')
    # loaded_model_AND.predictFun(X=[[1,1]])

    #Save Plot
    save_plot(df,model,filename=plotName)
