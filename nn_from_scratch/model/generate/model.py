import importlib
import os
import time

import numpy as np
import tensorflow as tf
import wandb
import yaml
from omegaconf import OmegaConf
from tensorflow.python.profiler import model_analyzer, option_builder
from wandb.keras import WandbCallback


class ModelSupervisor():
    def __init__(self, config_path=None):
        super().__init__()

        # default configs
        self.denses_params = [16]             # each element is the number of neurons of a dense layer
        self.activation = "relu"
        self.epochs = 10
        self.batch_size = 32
        self.dataset_info = {
            "name": "sinus",
            "path": "nn_from_scratch/model/generate/data/sinus/data.py",
            "args": {
                "n_samples": 1000,
                "test_ratio": 0.2,
                "random_seed": 42,
            }
        }
        self.random_seed = 42

        self.learning_rate = 1e-3
        self.fine_tuning_learning_rate = 1e-4
        self.fine_tuning_epochs = 5
        self.fine_tuning_batch_size = 128

        # update configs from the config file
        if config_path is not None:
            self.set_configs_from_file(config_path)

        # load corresponding dataset
        spec = importlib.util.spec_from_file_location("imported_module", self.dataset_info["path"])
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)

        # load data and create model
        self.dataset = imported_module.DatasetSupervisor(**self.dataset_info["args"])

        input_shape = self.dataset.feature_shape
        assert len(input_shape) == 1, "The input shape must be 1-dimensional."

        output_size = self.dataset.num_labels
        output_activation = self.dataset.output_activation
        self.model = self.create_model(input_shape, self.denses_params, output_size, self.activation, output_activation, self.random_seed)


    def set_configs_from_file(self, path):
        cfg = OmegaConf.load(path)
        cfg = OmegaConf.to_container(cfg, resolve=True)

        if "denses_params" in cfg:
            self.denses_params = cfg["denses_params"]
        if "activation" in cfg:
            self.activation = cfg["activation"]
        if "epochs" in cfg:
            self.epochs = cfg["epochs"]
        if "batch_size" in cfg:
            self.batch_size = cfg["batch_size"]
        if "dataset" in cfg:
            self.dataset_info = cfg["dataset"]
        if "random_seed" in cfg:
            self.random_seed = cfg["random_seed"]
        if "learning_rate" in cfg:
            self.learning_rate = cfg["learning_rate"]
        if "fine_tuning_learning_rate" in cfg:
            self.fine_tuning_learning_rate = cfg["fine_tuning_learning_rate"]
        if "fine_tuning_epochs" in cfg:
            self.fine_tuning_epochs = cfg["fine_tuning_epochs"]
        if "fine_tuning_batch_size" in cfg:
            self.fine_tuning_batch_size = cfg["fine_tuning_batch_size"]


    @staticmethod
    def create_model(input_shape, denses_params, output_size, activation, output_activation, random_seed=None):
        """
        Creates the model.

        Args:
            input_shape (tuple): The shape of the input.
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

        model.add(layers.Input(shape=input_shape))

        for i in range(len(denses_params)):
            n = denses_params[i]
            model.add(layers.Dense(n, activation=activation))

        model.add(layers.Dense(output_size, activation=output_activation))

        return model


    def compile_model(self, fine_tuning=False):
        if not fine_tuning:
            learning_rate = self.learning_rate
        else:
            learning_rate = self.fine_tuning_learning_rate

        self._compile_model(learning_rate, self.dataset.loss_function, self.dataset.metrics)


    def _compile_model(self, learning_rate, loss_function, metrics):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss=loss_function, metrics=metrics)


    def train_model(self, fine_tuning=False, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False):
        if not fine_tuning:
            epochs, batch_size, random_seed = self.epochs, self.batch_size, self.random_seed
        else:
            epochs, batch_size, random_seed = self.fine_tuning_epochs, self.fine_tuning_batch_size, self.random_seed

        return self._train_model(epochs, batch_size, tensorboard_log_dir, best_weights_dir, use_wandb, random_seed)


    def _train_model(self, epochs, batch_size, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False, random_seed=42):
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
        if self.dataset.test_x is None or self.dataset.test_y is None:
            validation_data = None
        else:
            validation_data = (self.dataset.test_x, self.dataset.test_y)

        callbacks = self._get_training_callbacks(tensorboard_log_dir, best_weights_dir, use_wandb)
        return self.model.fit(self.dataset.train_x, self.dataset.train_y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)


    def evaluate_model(self):
        return self._evaluate_model()


    def _evaluate_model(self):
        eval_vals = self.model.evaluate(self.dataset.test_x, self.dataset.test_y, verbose=0)
        eval_dict = {self.dataset.loss_function: eval_vals[0]}
        for i in range(len(self.dataset.metrics)):
            eval_dict[self.dataset.metrics[i]] = eval_vals[i+1]
        return eval_dict


    def get_model_info(self):
        return self._get_model_info()


    def _get_model_info(self):
        model_info = {
            "denses_params": self.denses_params,
            "activation": self.activation,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "fine_tuning_epochs": self.fine_tuning_epochs,
            "fine_tuning_batch_size": self.fine_tuning_batch_size,
            "fine_tuning_learning_rate": self.fine_tuning_learning_rate,
            "dataset": self.dataset_info,
            "random_seed": self.random_seed
        }
        return model_info


    def get_params_count(self):
        """
        Returns the number of parameters in the model.

        Returns:
            list[int]: The total number of parameters, the number of trainable parameters, and the number of non-trainable parameters.
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for layer in self.model.variables:
            total_params += np.prod(layer.shape)

        for layer in self.model.trainable_variables:
            trainable_params += np.prod(layer.shape)

        non_trainable_params = total_params - trainable_params

        return int(total_params), int(trainable_params), int(non_trainable_params)


    def get_FLOPs(self):
        """
        Returns the number of FLOPs of the model.

        Returns:
            int: The number of FLOPs.
        """
        input_signature = [
            tf.TensorSpec(
                shape=(1, *params.shape[1:]),
                dtype=params.dtype,
                name=params.name
            ) for params in self.model.inputs
        ]
        forward_graph = tf.function(self.model, input_signature).get_concrete_function().graph
        options = option_builder.ProfileOptionBuilder.float_operation()
        options['output'] = 'none'
        graph_info = model_analyzer.profile(forward_graph, options=options)

        FLOPs = graph_info.total_float_ops

        return FLOPs


    def measure_execution_time(self):
        """
        Measures the execution time of the model.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            float: The execution time in ms.
        """
        rng = np.random.RandomState(42)
        sample_idx = rng.randint(0, self.dataset.train_x.shape[0])
        x = np.array([self.dataset.train_x[sample_idx]])

        # warm up
        tic = time.time()
        for i in range(100):
            self.model(x, training=False)
        toc = time.time()
        itr = int(10 * 100 / (toc - tic))

        # run the test
        tic = time.time()
        for i in range(itr):
            self.model(x, training=False)
        toc = time.time()
        execution_time = (toc-tic)/itr*1000     # in ms

        return execution_time


    def save_eqcheck_data(self, n_samples, save_dir):
        """
        Saves the eqcheck data as {"data_x", "data_y_pred"}

        The data_x has shape (samples, *input_shape) and data_y_pred has shape (samples, *output_shape).

        Args:
            n_samples (int): The number of samples to be saved
            save_dir (str): The directory where the data should be saved
        """
        # sanity check
        for data_dim_size, model_dim_size in zip(self.dataset.train_x.shape[1:], self.model.inputs[0].shape[1:]):
            if data_dim_size != model_dim_size and model_dim_size is not None:
                raise ValueError("The shape of the train_x doesn't match the input shape of the model.")

        data_x = self.dataset.train_x[:n_samples]
        data_y_pred = self.model.predict(data_x, verbose=0)

        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'eqcheck_data.npz'), data_x=data_x, data_y_pred=data_y_pred)


    def save_model(self, save_dir):
        """
        Saves the model to the specified directory.

        Args:
            save_dir (str): The directory where the model should be saved in.
        """
        # save the model as a Keras format
        os.makedirs(os.path.join(save_dir, "keras_format"), exist_ok=True)
        self.model.save(os.path.join(save_dir, "keras_format/model.keras"))

        # save the model as a SavedModel format
        os.makedirs(os.path.join(save_dir, "saved_model_format"), exist_ok=True)
        tf.saved_model.save(self.model, os.path.join(save_dir, "saved_model_format"))


    @staticmethod
    def log_model_to_wandb(model_dir, model_save_name):
        """
        Logs the model to wandb.

        Args:
            save_dir (str): The directory where the model is stored.
            model_save_name (str): The name that will be assigned to the model artifact.
        """
        model_path = os.path.join(model_dir, "keras_format/model.keras")
        model_artifact = wandb.Artifact(model_save_name, type="model")
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)


    def save_weights(self, save_dir):
        """
        Saves the model weights to the specified directory.

        Args:
            save_dir (str): The directory where the model weights should be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(os.path.join(save_dir, "weights.weights.h5"))


    def load_weights(self, load_dir):
        """
        Loads the model weights from the specified directory.

        Args:
            load_dir (str): The directory where the model weights are stored.
        """
        self.model.load_weights(os.path.join(load_dir, "weights.weights.h5"))


    @staticmethod
    def save_model_info(model_info, save_dir):
        """
        Saves the model info to the specified directory.

        Args:
            model_info (dict): The model info.
            save_dir (str): The directory where the model info should be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        yaml.Dumper.ignore_aliases = lambda *args : True
        with open(os.path.join(save_dir, "model_info.yaml"), 'w') as f:
            yaml.dump(model_info, f, indent=4, sort_keys=False)


    def _get_training_callbacks(self, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False):
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
