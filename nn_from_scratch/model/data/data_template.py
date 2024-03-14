"""
This module contains a template class that other datasets should inherit from.
"""


class DatasetSupervisorTemplate:
    """
    This class is a template for datasets. In order to create a new dataset, you should
    inherit from this class and implement its abstract functions.
    """

    def __init__(self, **kwargs):
        """
        Initializes the class by setting the following attributes:
            self.train_x (numpy.ndarray): The training data.
            self.train_y (numpy.ndarray): The training labels.
            self.test_x (numpy.ndarray): The test data.
            self.test_y (numpy.ndarray): The test labels.

            self.feature_shape (tuple): The shape of the features.
            self.num_labels (int): The number of labels.
            self.output_activation (str): The name of the activation function of the output layer.
            self.loss_function (str | tf.keras.losses.Loss): The name of the loss function or the loss function itself.
            self.metrics (list): The list of metrics.
        """
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.feature_shape = None
        self.num_labels = None
        self.output_activation = None
        self.loss_function = None
        self.metrics = None


    def load_dataset(**kwargs):
        """
        Loads the dataset and returns it in the format ((trainX, trainY), (testX, testY)).
        The data should be in numpy float32 and shuffled.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        raise NotImplementedError
