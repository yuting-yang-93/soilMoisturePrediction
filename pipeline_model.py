from configuration import ModelConfiguration
config = ModelConfiguration()
from datetime import datetime
import os.path
from datetime import datetime
import pandas as pd
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from model.generate3DShape import generated3D
from hyperParGraph.lossGraph import LossHistory
from model.LSTM_model import  log_dir_name, create_model
from model.generateForecast import generateLSTMForecast
from model.generateForecast import generateLRForecast

data = pd.read_csv('../data/preparedData.csv')
targetVar = ['sm_20']
# normal Parameters
timesteps = config.timesteps
batch_size = 8*8
epochs = 200
localPath = 'model_LSTM_20cm'
if not os.path.isdir('./%s'%localPath):
    os.makedirs('./%s'%localPath)


#===============  LSTM model
inputCols = ['sm_5', 'p', 'st_5', 'at', 'station', 'TAXNWRB_Class1', 'TAXOUSDA_Class1']
X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, test, test_scaled, scaler_y = \
    generated3D (data, targetVar,inputCols,isScaled=True,isDummy=False)
input_dim = X_train_scaled.shape[2]
output_dim = len(targetVar)


#hyper-parameters
dim_learning_rate = Categorical(categories=[0.00001, 0.0001, 0.001, 0.005, 0.01,],name='learning_rate')
dim_num_dense_layers = Integer(low = 0, high = 3, name = 'num_dense_layers')
dim_num_dense_first_node = Categorical(categories=[8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512],name='num_dense_fisrt_node')
dim_num_dense_nodes = Categorical(categories=[8, 16, 32, 64, 128, 96, 128, 192, 256, 320, 384, 448, 512],name='num_dense_nodes')
dim_dropout_rate = Categorical(categories=[0, 0.1, 0.2, 0.3, 0.4, 0.5],name='dropout_rate')
dim_first_activation = Categorical(categories=['tanh', 'linear', 'sigmoid'],name='first_activation')
dim_activation = Categorical(categories=['relu', 'tanh'],name='activation')

dimensions = [dim_learning_rate, dim_num_dense_layers, dim_num_dense_first_node,
              dim_num_dense_nodes, dim_dropout_rate, dim_first_activation, dim_activation]


# needed: train_scaled, val_scaled, epoch, steps_per_epoch, batch_size, timesteps, input_dim, output_dim
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_fisrt_node, num_dense_nodes, dropout_rate, first_activation, activation):

    model = create_model(timesteps, input_dim, output_dim,
                         learning_rate, num_dense_layers, num_dense_fisrt_node, num_dense_nodes,
                         dropout_rate, first_activation, activation)

    #=========== CallBack
    today = datetime.now()
    time = today.strftime('%Y.%m.%d.%H_%M')

    # Loss Graph Name
    s  = '{0}_{1}_layers{2}_nodes[{3},{4}]_dropout{5}_[{6},{7}]'

    modelName = s.format(time, learning_rate, num_dense_layers, num_dense_fisrt_node, num_dense_nodes,
                         dropout_rate, first_activation, activation)
    historyGraph = LossHistory(modelName, localPath)


    # Log file Path

    logName = './%s/logs'%localPath
    log_dir = log_dir_name(learning_rate, num_dense_layers, num_dense_fisrt_node,
                           num_dense_nodes, dropout_rate, first_activation, activation, logName)
    callback_tensorboard = TensorBoard(log_dir = log_dir,
                                       histogram_freq = 0,
                                       write_graph = True)

    # best model weigths
    weights_path = './%s/weights' % localPath

    if not os.path.isdir(weights_path):
        os.makedirs (weights_path)

    filepath="{0}/weights_{1}.keras".format(weights_path, modelName)
    callback_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                          mode='min' ,save_best_only='True', verbose = 1, period = 1)

    # early stop
    callback_early_stopping = EarlyStopping(monitor = 'val_loss', patience=30, verbose=1, min_delta=0.0001)

    callbacks = [historyGraph,
                 callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard]

    history = model.fit(X_train_scaled, y_train_scaled,
                        batch_size=batch_size, epochs=epochs,
                        verbose=1,
                        validation_data=(X_val_scaled, y_val_scaled),
                        callbacks=callbacks)

    path_best_model = './%s/best_model.keras'%localPath
    # path_best_model = './best_model.keras'
    mae = history.history['val_loss'][-1]

    # Print the mean absolute error
    print('mean absolute error:', mae)
    print()

    # default a relative high value of best_mae
    best_mae = 1000

    log_path = './%s/savedModel'%localPath
    if not os.path.isdir(log_path):
        os.makedirs (log_path)

    #  Insert all the hyper-parameters in the dir-name.
    log_dir="{0}/model_{1}.keras".format(log_path, modelName)

    model.save(log_dir)

    # If the mae of the saved model is improved...
    if mae < best_mae:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the mae
        best_mae = mae

    historyGraph.loss_plot('epoch')

    del model
    K.clear_session()
    return mae


default_parameters = [0.001, 0, 128, 32, 0,  'sigmoid', 'relu']
# Normal Fitting
fitness(x = default_parameters)

# Baysian Optimization
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=15,
                            x0=default_parameters)


# generate the LSTM forecast
# After fitting model, the best model (keras file) for one combination hyper-parameter is stored in "./weights/" file
weigthKerasfile = ''
path_best_model = './{0}/weights/{1}.keras'.format(localPath, weigthKerasfile)
model = load_model(path_best_model)
isStandard = True
forecastLSTM = generateLSTMForecast (model, scaler_y, test, test_scaled, timesteps, output_dim, overlappedSize=config.overlappedSize, isStandard = True)
forecastLSTM = forecastLSTM.rename(columns={"forecast": "LSTM", 'AE': "AE_LSTM"})


# generate the LR forecast
inputCols = ['sm_5', 'p', 'st_5', 'at', 'station', 'TAXNWRB_Class1', 'TAXOUSDA_Class1']
X_train, y_train, X_val, y_val, test, test_scaled, scaler_y = \
    generated3D (data, targetVar,inputCols,isScaled=False, isDummy=True)
forecastLR = generateLRForecast (X_train, y_train, test, timesteps, output_dim)
forecastLR = forecastLR.rename(columns={"forecast": "fore_lr", 'AE': "AE_lr"})

# generate the LR forecast
if targetVar[0] == "sm_20":
    inputCols = ['sm_5_a94', 'p_a94', 'st_5_a94', 'at_a94', 'station', 'TAXNWRB_Class1', 'TAXOUSDA_Class1']
else:
    inputCols = ['sm_5_a99', 'p_a99', 'st_5_a99', 'at_a99', 'station', 'TAXNWRB_Class1', 'TAXOUSDA_Class1']

X_train, y_train, X_val, y_val, test, test_scaled, scaler_y = \
    generated3D (data, targetVar,inputCols,isScaled=False, isDummy=True)
forecastEWLR = generateLRForecast (X_train, y_train, test, timesteps, output_dim)
forecastEWLR = forecastEWLR.rename(columns={"forecast": "fore_ewlr", 'AE': "AE_ewlr"})