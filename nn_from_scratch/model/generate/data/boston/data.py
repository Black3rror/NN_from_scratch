import numpy as np
import tensorflow as tf

from nn_from_scratch.model.generate.data.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, modify_dataset=False, noise=5, shift=5, random_seed=None):
        """
        Initializes the class.

        Args:
            modify_dataset (bool): If True, the trainset will be modified. The modification parameters are `noise` and `shift`.
            noise (float): The noise level that should be added as modification.
            shift (float): The shift value that should be added as modification.
            random_seed (int): The random seed. If None, the random seed is not set.
        """
        super().__init__()

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset()

        if modify_dataset:
            rng = np.random.RandomState(random_seed)
            delta = rng.rand(*self.train_y.shape) * noise + shift
            self.train_y += delta

        self.feature_size = 13
        self.num_labels = 1
        self.output_activation = "linear"
        self.loss_function = "mse"
        self.metrics = ["mae"]


    @staticmethod
    def load_dataset():
        """
        Loads the dataset.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        # train_x, test_x will have shape (num_samples, 13) and dtype float64
        # train_y, test_y will have shape (num_samples,) and dtype float64
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.boston_housing.load_data()

        # normalize data
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)

        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)

        return (train_x, train_y), (test_x, test_y)
