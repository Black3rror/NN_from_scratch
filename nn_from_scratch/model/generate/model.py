import os
import time

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.python.profiler import model_analyzer, option_builder   # type: ignore
from wandb.keras import WandbCallback


def create_model(input_size, denses_params, output_size, activation, output_activation, random_seed=None):
    """
    Creates the model.

    Args:
        input_size (int): The size of the input.
        denses_params (list): Each element is the number of neurons of a dense layer (excluding the output layer).
        output_size (int): The size of the output.
        activation (str): The activation function of the hidden layers.
        output_activation (str): The activation function of the output layer.
        random_seed (int): The random seed.
    """
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    model = tf.keras.Sequential()
    layers = tf.keras.layers

    model.add(layers.Input(shape=(input_size,)))

    for i in range(len(denses_params)):
        n = denses_params[i]
        model.add(layers.Dense(n, activation=activation))

    model.add(layers.Dense(output_size, activation=output_activation))

    return model


def train_model(model, train_x, train_y, epochs, batch_size, test_x=None, test_y=None, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False, random_seed=42):
    """
    Trains the model.

    Args:
        model (tf.keras.Model): The model.
        train_x (numpy.ndarray): The input data for training.
        train_y (numpy.ndarray): The output data for training.
        epochs (int): The number of epochs.
        batch_size (int): The batch size.
        test_x (numpy.ndarray, optional): The input data for testing. If None, test validation will be ignored. Defaults to None.
        test_y (numpy.ndarray, optional): The output data for testing. If None, test validation will be ignored. Defaults to None.
        tensorboard_log_dir (str, optional): The directory where the logs should be saved. If None, the logs won't be saved. Defaults to None.
        best_weights_dir (str, optional): The directory where the best weights should be saved. If None, the best weights won't be saved. Defaults to None.
        use_wandb (bool, optional): If True, the training progress will be logged to wandb. Defaults to False.
        random_seed (int, optional): The random seed. Defaults to 42.
    """
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)
    if test_x is None or test_y is None:
        validation_data = None
    else:
        validation_data = (test_x, test_y)

    callbacks = _get_training_callbacks(tensorboard_log_dir, best_weights_dir, use_wandb)
    return model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)


def get_params_count(model):
    """
    Returns the number of parameters in the model.

    Args:
        model (tf.keras.Model): The model.

    Returns:
        list[int]: The total number of parameters, the number of trainable parameters, and the number of non-trainable parameters.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for layer in model.variables:
        total_params += np.prod(layer.shape)

    for layer in model.trainable_variables:
        trainable_params += np.prod(layer.shape)

    non_trainable_params = total_params - trainable_params

    return int(total_params), int(trainable_params), int(non_trainable_params)


def get_FLOPs(model):
    """
    Returns the number of FLOPs of the model.

    Args:
        model (tf.keras.Model): The model.

    Returns:
        int: The number of FLOPs.
    """
    input_signature = [
        tf.TensorSpec(
            shape=(1, *params.shape[1:]),
            dtype=params.dtype,
            name=params.name
        ) for params in model.inputs
    ]
    forward_graph = tf.function(model, input_signature).get_concrete_function().graph
    options = option_builder.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'
    graph_info = model_analyzer.profile(forward_graph, options=options)

    FLOPs = graph_info.total_float_ops

    return FLOPs


def measure_execution_time(model, x):
    """
    Measures the execution time of the model.

    Args:
        model (tf.keras.Model): The model.
        x (numpy.ndarray): The input data.

    Returns:
        float: The execution time in ms.
    """
    # warm up
    tic = time.time()
    for i in range(100):
        model(x, training=False)
    toc = time.time()
    itr = int(10 * 100 / (toc - tic))

    # run the test
    tic = time.time()
    for i in range(itr):
        model(x, training=False)
    toc = time.time()
    execution_time = (toc-tic)/itr*1000     # in ms

    return execution_time


def save_model(model, save_dir):
    """
    Saves the model to the specified directory.

    Args:
        model (tf.keras.Model): The model.
        save_dir (str): The directory where the model should be saved in.
    """
    # save the model as a Keras format
    os.makedirs(os.path.join(save_dir, "keras_format"), exist_ok=True)
    model.save(os.path.join(save_dir, "keras_format/model.keras"))

    # save the model as a SavedModel format
    os.makedirs(os.path.join(save_dir, "saved_model_format"), exist_ok=True)
    tf.saved_model.save(model, os.path.join(save_dir, "saved_model_format"))


@staticmethod
def log_model_to_wandb(model_dir, model_save_name):
    """
    Logs the model to wandb.

    Args:
        model_dir (str): The directory where the model is saved.
        model_save_name (str): The name that will be assigned to the model artifact.
    """
    model_path = os.path.join(model_dir, "keras_format/model.keras")
    model_artifact = wandb.Artifact(model_save_name, type="model")
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)


def save_weights(model, save_dir):
    """
    Saves the model weights to the specified directory.

    Args:
        model (tf.keras.Model): The model.
        save_dir (str): The directory where the model weights should be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_weights(os.path.join(save_dir, "weights.weights.h5"))


def _get_training_callbacks(tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False):
    """
    Returns the training callbacks.

    Args:
        tensorboard_log_dir (str, optional): The directory where the logs should be saved. If None, the logs won't be saved. Defaults to None.
        best_weights_dir (str, optional): The directory where the best weights should be saved. If None, the best weights won't be saved. Defaults to None.
        use_wandb (bool, optional): If True, the training progress will be logged to wandb. Defaults to False.
    """
    callbacks = []

    if tensorboard_log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1))

    if best_weights_dir is not None:
        best_weights_path = os.path.join(best_weights_dir, "best.weights.h5")
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_weights_path, save_best_only=True, save_weights_only=True, verbose=0))

    if use_wandb and False:     # wandb callback seems to raise error in tf 2.16.1
        callbacks.append(WandbCallback(save_model=False, log_weights=True, compute_flops=True))

    return callbacks
