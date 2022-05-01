from utils.all_utils import Data_prepare, save_plot, main

# creating entry point for code
if __name__ == '__main__':
# Data creation & preperation
    AND= {
        'x1':[0,0,1,1],
        'x2':[0,1,0,1],
        'y':[0,0,0,1]
    }
    ETA = 0.1
    EPOCHS = 10

    main(data=AND, modelName='model_AND',plotName='model_AND.png',eta=ETA,epochs=EPOCHS)

