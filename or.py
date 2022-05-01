import pandas as pd
from utils.all_utils import Data_prepare, save_plot
from utils.model import Perceptron

#defining main function
def main(data, modelName,plotName,eta,epochs):
    df_OR = pd.DataFrame(data)
    X,y = Data_prepare(df_OR)

    # Model initialising & Fittign
    model_OR = Perceptron(eta=eta,epochs=epochs)
    model_OR.fit(X,y)

    # Save,Load& Predict from model
    model_OR.SaveModel(filename=modelName)
    # loaded_model_AND = Perceptron().loadModel_(filepath='/content/model/model_AND')
    # loaded_model_AND.predictFun(X=[[1,1]])

    #Save Plot
    save_plot(df_OR,model_OR,filename=plotName)

# creating entry point for code
if __name__ == '__main__':
# Data creation & preperation
    OR= {
        'x1':[0,0,1,1],
        'x2':[0,1,0,1],
        'y':[0,1,1,1]
    }
    ETA = 0.1
    EPOCHS = 10

    main(data=OR, modelName='model_OR',plotName='model_OR.png',eta=ETA,epochs=EPOCHS)

