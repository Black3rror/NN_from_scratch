import numpy as np

from nn_from_scratch.model.generate.data.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, n_samples, test_ratio, random_seed=None):
        super().__init__()

        self.n_samples = n_samples
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(n_samples, test_ratio, random_seed)

        self.feature_shape = self.train_x.shape[1:]
        self.num_labels = 1
        self.output_activation = "linear"
        self.loss_function = "mse"
        self.metrics = ["mae"]


    @staticmethod
    def load_dataset(n_samples, test_ratio, random_seed=None):
        """
        Loads the dataset.

        Args:
            n_samples (int): The number of samples that should be in the whole dataset (train + test).
            test_ratio (float): The ratio of the test set.
            random_seed (int): The random seed. If None, the random seed is not set.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        dataset_x, dataset_y = DatasetSupervisor._create_dataset(n_samples, random_seed)
        dataset_x = dataset_x.astype(np.float32)
        dataset_y = dataset_y.astype(np.float32)

        rng = np.random.RandomState(random_seed)
        shuffled_indices = rng.permutation(len(dataset_x))

        dataset_x = dataset_x[shuffled_indices]
        dataset_y = dataset_y[shuffled_indices]

        test_size = int(len(dataset_x)*test_ratio)
        train_x = dataset_x[test_size:]
        train_y = dataset_y[test_size:]
        test_x = dataset_x[:test_size]
        test_y = dataset_y[:test_size]

        return (train_x, train_y), (test_x, test_y)


    @staticmethod
    def _create_dataset(n_samples, random_seed=None):
        rng = np.random.RandomState(random_seed)
        x = rng.uniform(0, 2*np.pi, n_samples)
        y = np.sin(x)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return x, y
