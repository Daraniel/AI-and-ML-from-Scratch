import tempfile

import numpy as np
import pytest

from common.exceptions import (InvalidArgumentException,
                               ModelNotTrainedException)
from common.utils import ActivationFunctionsForNN
from models.neural_networks import (ConvolutionalLayer, FlattenLayer,
                                    LinearLayer, NeuralNetworkModel,
                                    PoolingLayer)


class TestNeuralNetworkModel:
    @pytest.fixture
    def simple_neural_network(self):
        model = NeuralNetworkModel()
        model.add_layer(FlattenLayer())
        model.add_layer(LinearLayer(input_shape=10, output_shape=5))
        model.add_layer(LinearLayer(input_shape=5, output_shape=2))
        return model

    def test_neural_network_creation(self, simple_neural_network):
        assert isinstance(simple_neural_network, NeuralNetworkModel)

    def test_add_layer_invalid_type(self, simple_neural_network):
        with pytest.raises(InvalidArgumentException):
            simple_neural_network.add_layer("invalid_layer")

    def test_add_layer_valid_type(self, simple_neural_network):
        model = NeuralNetworkModel()
        model.add_layer(
            ConvolutionalLayer(
                kernel_size=3, padding=0, stride=1, input_channels=1, output_channels=10
            )
        )
        assert isinstance(model.layers[0], ConvolutionalLayer)

    def test_add_linear_layer_as_first_layer(self, simple_neural_network):
        model = NeuralNetworkModel()
        model.add_layer(LinearLayer(input_shape=10, output_shape=5))
        # When using LinearLayer as the first layer, a FlattenLayer layer is created first due to implementation details
        assert isinstance(model.layers[0], FlattenLayer)
        assert isinstance(model.layers[1], LinearLayer)

    def test_learn(self, simple_neural_network):
        x = np.random.rand(10, 10)
        y = np.random.rand(10, 2)
        costs = simple_neural_network.learn(x, y, epochs=2, batch_size=x.shape[0])
        assert simple_neural_network.is_trained
        assert len(costs) == 2

    def test_not_trained(self, simple_neural_network):
        x = np.random.rand(10, 10)
        with pytest.raises(ModelNotTrainedException):
            simple_neural_network.infer_inputs(x)

    def test_forward_pass_simple_model(self, simple_neural_network):
        x = np.random.rand(10, 10)
        y = np.random.rand(10, 2)
        simple_neural_network.learn(x, y, epochs=1, batch_size=5, verbose=False)
        results = simple_neural_network.infer_inputs(x)
        assert results.shape == y.shape

    def test_infer_complex_model(self):
        X = np.random.rand(10, 28, 28, 1)
        y = np.random.rand(10, 10)
        LEARNING_RATE = 0.1
        LAMBDA = 0.01
        EPOCHS = 1

        cnn = NeuralNetworkModel()

        layer1 = ConvolutionalLayer(
            kernel_size=3,
            padding=0,
            stride=1,
            input_channels=1,
            output_channels=10,
            learning_rate=LEARNING_RATE,
            lambda_regularization=LAMBDA,
        )  # -> 26x26x10
        layer2 = PoolingLayer(stride=2, use_mean_pooling=True)  # -> 13x13x10
        layer3 = ConvolutionalLayer(
            kernel_size=5,
            padding=0,
            stride=1,
            input_channels=10,
            output_channels=16,
            learning_rate=LEARNING_RATE,
            lambda_regularization=LAMBDA,
        )  # -> 9x9x16
        layer4 = FlattenLayer()  # size is automatically calculated -> mx1296
        layer5 = LinearLayer(
            input_shape=1296,
            output_shape=100,
            learning_rate=LEARNING_RATE,
            lambda_regularization=LAMBDA,
        )  # -> mx100
        layer6 = LinearLayer(
            input_shape=100,
            output_shape=10,
            learning_rate=LEARNING_RATE / 10,
            activation_function=ActivationFunctionsForNN.Softmax(),
            lambda_regularization=LAMBDA,
        )  # -> mx10

        cnn.add_layer(layer1)
        cnn.add_layer(layer2)
        cnn.add_layer(layer3)
        cnn.add_layer(layer4)
        cnn.add_layer(layer5)
        cnn.add_layer(layer6)

        cnn.learn(X, y, EPOCHS, 64)
        results = cnn.infer_inputs(X)
        assert results.shape == y.shape

    def test_save_and_load_weights(self, simple_neural_network):
        x = np.random.rand(10, 10)
        y = np.random.rand(10, 2)

        simple_neural_network.learn(x, y, epochs=1, batch_size=x.shape[0])
        weights_before = [
            layer.weights.copy()
            for layer in simple_neural_network.layers
            if hasattr(layer, "weights")
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = temp_dir + "/test_weights.pkl"
            simple_neural_network.save_weights(file_path)

            loaded_model = NeuralNetworkModel()
            loaded_model.add_layer(LinearLayer(input_shape=10, output_shape=5))
            loaded_model.add_layer(LinearLayer(input_shape=5, output_shape=2))
            loaded_model.load_weights(file_path)

            weights_after = [
                layer.weights.copy()
                for layer in loaded_model.layers
                if hasattr(layer, "weights")
            ]
            assert len(weights_before) == len(weights_after)
            assert all(
                np.allclose(weights_before[i], weights_after[i])
                for i in range(len(weights_before))
            )
