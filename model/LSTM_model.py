# HyperParameter Tuning
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout

from configuration import ModelConfiguration
config = ModelConfiguration()

def log_dir_name(learning_rate, num_dense_layers, num_dense_fisrt_node, num_dense_nodes,
                 dropout_rate, first_activation, activation, logName):
    '''

    :param learning_rate: Float. Learning rate of model
    :param num_dense_layers: Int. The number of LSTM layers
    :param num_dense_fisrt_node: Int. The number of nodes for first layer
    :param num_dense_nodes: Int. The number of nodes for rest layers
    :param dropout_rate:  Float. droppout rate
    :param first_activation: String. Activation of the first layer
    :param activation: String. Activation of the rest layers
    :param logName: String. The path name of log.
    :return: log_dir: log file of tensorboard
    '''


    today = datetime.now()
    time = today.strftime('%Y.%m.%d.%H:%M')

    # The dir-name for the TensorBoard log-dir:
    s  = './%s/{0}_{1}_{2}layers_[{3},{4}]nodes_{5}dropout_[{6},{7}]/'%logName

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(time, learning_rate, num_dense_layers, num_dense_fisrt_node, num_dense_nodes,
                       dropout_rate, first_activation, activation)
    return log_dir



def loss_mae_warumup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model_all's output.
    """
    warmup_steps = config.warmup_steps
    # The shape of both input tensors are:
    # [batch_size, timesteps, output_dim].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_absolute_error(y_true_slice, y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


def create_model(timesteps, input_dim, output_dim,
                 learning_rate, num_dense_layers, num_dense_fisrt_node, num_dense_nodes,
                 dropout_rate, first_activation, activation):
    '''

    :param timesteps: Int. The number of data points for one sequence.
    :param input_dim: Int. The number of input variables.
    :param output_dim: Int. The number of output variables.
    :param learning_rate: Float. Learning rate of model
    :param num_dense_layers: Int. The number of LSTM layers
    :param num_dense_fisrt_node: Int. The number of nodes for first layer
    :param num_dense_nodes: Int. The number of nodes for rest layers
    :param dropout_rate:  Float. droppout rate
    :param first_activation: String. Activation of the first layer
    :param activation: String. Activation of the rest layers
    :param logName: String. The path name of log.
    :return:
    '''

    # Define Model
    model = Sequential()
    model.add(LSTM(num_dense_fisrt_node, input_shape=(timesteps, input_dim), activation=first_activation, return_sequences=True))
    model.add(Dropout(dropout_rate))

    # iteration of layers
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+2)
        model.add(LSTM(num_dense_nodes, activation=activation,return_sequences=True, name =name))
        model.add(Dropout(dropout_rate))

    model.add(Dense(output_dim))
    optimizer = optimizers.Adam(lr = learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    print (model.summary())

    return model

