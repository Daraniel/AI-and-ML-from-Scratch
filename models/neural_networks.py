import abc
import pickle
from numbers import Number
from typing import List

import numpy as np
from tqdm import tqdm

from common.base_model import BaseModel
from common.exceptions import InvalidArgumentException, ModelNotTrainedException
from common.utils import EvaluationMetrics, ActivationFunctionsForNN


class NeuralNetworkModel(BaseModel):
    def __init__(self, cost_function=EvaluationMetrics.cross_entropy_loss):
        """
        create a neural network model
        :param cost_function: cos function used to evaluate the performance of the model
        """
        if not callable(cost_function):
            raise InvalidArgumentException(f'cost_function must be a callable not a {type(cost_function)}')

        self.cost_function = cost_function
        self.layers: List[BaseLayer]
        self.layers = []
        self.is_trained = False

    def get_number_of_parameters(self) -> int:
        """
        calculates and return total number of parameters in this model
        :return: number of parameters
        """
        number_of_parameters = 0
        for layer in self.layers:
            number_of_parameters += layer.get_number_of_parameters()
        return number_of_parameters

    def add_layer(self, layer):
        """
        adds a new layer to the end of the current neural network model
        :param layer: layer to add, must be an instance of BaseLayer
        """
        if not isinstance(layer, BaseLayer):
            raise InvalidArgumentException("Layer must inherit from BaseLayer")
        if len(self.layers) == 0 and isinstance(layer, LinearLayer):
            print("When using LinearLayer as the first layer,"
                  " a flattening operation must be applied to the data due to internal implementation details")
            self.layers.append(FlattenLayer())
        self.layers.append(layer)

    def _forward(self, x):
        activations = {0: x}
        weight_sums = {0: x}
        for i, layer in enumerate(self.layers):
            weighted_sum, x = layer.forward(x)
            activations[i + 1] = x
            weight_sums[i + 1] = weighted_sum
        return weight_sums, activations

    def _back_propagate(self, weight_sums, activations, y):
        delta = activations[len(self.layers)] - y
        gradients = {}
        gradients_bias = {}

        for i in range(len(self.layers), 0, -1):
            layer = self.layers[i - 1]
            if isinstance(layer, (LinearLayer, ConvolutionalLayer)):
                gradient, bias = layer.compute_gradients(delta, activations[i - 1])
                gradients[i - 1] = gradient
                gradients_bias[i - 1] = bias

            delta = layer.back_propagate(delta, weight_sums[i - 1])

        return gradients, gradients_bias

    def _gradient_decent(self, gradients, gradients_biases, number_of_records):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (LinearLayer, ConvolutionalLayer)):
                gradient = gradients[i]
                gradients_bias = gradients_biases[i]
                layer.gradient_decent(gradient, gradients_bias, number_of_records)

    def learn(self, x: np.array, y: np.array, epochs=100, batch_size=10, verbose=True):
        """
        train the model with the given input and labels
        :param x: input features
        :param y: labels
        :param epochs: number of epochs to train the model
        :param batch_size: batch size, if 1, the model will use stochastic gradient descent
        :param verbose: whether to print progress during training
        :return: list of costs of each epoch
        """
        if not isinstance(epochs, int) or epochs <= 0:
            raise InvalidArgumentException("epochs must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise InvalidArgumentException("batch_size must be a positive integer")

        if batch_size == 1:  # use stochastic gradient descent
            costs = []
            m = len(x)
            for epoch in range(epochs):
                for i in tqdm(np.random.permutation(m), disable=not verbose):
                    weight_sums, activations = self._forward(np.atleast_3d(x[i]))
                    gradients, gradients_bias = self._back_propagate(weight_sums, activations, y[i])
                    self._gradient_decent(gradients, gradients_bias, 1)

                self.is_trained = True
                cost = self.calculate_cost(x, y)
                if verbose:
                    print(f'Cost at epoch {epoch} is {cost}')
                costs.append(cost)

        else:  # use normal gradient descent
            costs = []
            m = len(x)
            for epoch in range(epochs):
                batches = self._get_batch(m, batch_size)
                for batch in tqdm(batches, total=m // batch_size + 1, disable=not verbose):
                    first_gradient, first_gradient_bias = 0, 0
                    for element in batch:
                        weight_sums, activations = self._forward(np.atleast_3d(x[element]))
                        gradients, gradients_bias = self._back_propagate(weight_sums, activations, y[element])

                        if element == batch[0]:
                            first_gradient = gradients
                            first_gradient_bias = gradients_bias
                        else:
                            first_gradient = dict(
                                [(k, first_gradient[k] + gradients[k]) for k in first_gradient.keys()])
                            first_gradient_bias = dict(
                                [(k, first_gradient_bias[k] + gradients_bias[k]) for k in first_gradient_bias.keys()])

                    self._gradient_decent(first_gradient, first_gradient_bias, batch_size)

                self.is_trained = True
                cost = self.calculate_cost(x, y)
                if verbose:
                    print(f'Cost at epoch {epoch} is {cost}')
                costs.append(cost)

        return np.array(costs)

    @staticmethod
    def _get_batch(n, batch_size):
        start_index = 0
        idx = np.random.permutation(n)
        while True:
            stop_index = min(start_index + batch_size, n)
            i = idx[start_index:stop_index]
            if start_index > n - 1:
                break
            yield i
            start_index += batch_size

    def calculate_cost(self, x, y):
        """
        calculates the cost of predicting the labels from input features
        :param x: input features
        :param y: labels
        :return: cost of the prediction
        """
        cost = 0
        for i in range(len(x)):
            yhat = self.infer(x[i])
            cost += self.cost_function(yhat, y[i])

        for layer in self.layers:
            if hasattr(layer, 'filters'):
                cost += np.sum([np.sum(ii ** 2) for ii in layer.filters])
            elif hasattr(layer, 'weights'):
                cost += np.sum(layer.weights ** 2)

        return cost / len(x)

    def infer(self, x: np.array) -> np.array:
        """
        predicts the labels of a single input
        :param x: input features
        :return: label of input
        """
        if not self.is_trained:
            raise ModelNotTrainedException("Please train the model before using it")

        x = np.atleast_3d(x)
        weight_sums, activations = self._forward(x)
        yhat = activations[len(self.layers)]
        return yhat

    def infer_inputs(self, inputs_features: np.array) -> np.array:
        """
        predicts the labels of inputs
        :param inputs_features: features of inputs
        :return: predicted label of inputs
        """
        return np.array([self.infer(features) for features in inputs_features])

    def save_weights(self, file_name):
        """
        save the weights of the model to the given file
        :param file_name: name of the file to save the weights in
        """
        parameters_dictionary = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvolutionalLayer):
                parameters = {'learning_rate': layer.learning_rate,
                              'lambda_regularization': layer.lambda_regularization,
                              'filters': layer.filters,
                              'biases': layer.biases}
            elif isinstance(layer, LinearLayer):
                parameters = {'learning_rate': layer.learning_rate,
                              'lambda_regularization': layer.lambda_regularization,
                              'weights': layer.weights,
                              'biases': layer.biases}
            else:
                continue

            parameters_dictionary[str(i)] = parameters

        with open(file_name, 'wb') as file:
            pickle.dump(parameters_dictionary, file)

    def load_weights(self, file_name):
        """
        loads the weights for the model from the given file, please note that the layers need to be added manually
        before loading the weights
        :param file_name: name of the file to load the weights from
        """
        with open(file_name, 'rb') as file:
            parameters_dictionary = pickle.load(file)
            for i, layer in enumerate(self.layers):
                if str(i) in parameters_dictionary:
                    parameters = parameters_dictionary[str(i)]
                    layer.learning_rate = parameters['learning_rate']
                    layer.lambda_regularization = parameters['lambda_regularization']
                    layer.biases = parameters['biases']
                    if isinstance(layer, LinearLayer):
                        layer.weights = parameters['weights']
                    elif isinstance(layer, ConvolutionalLayer):
                        layer.filters = parameters['filters']


class BaseLayer(abc.ABC):
    """
    base neural network layer, all layers must inherit from this layer
    """
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def back_propagate(self, delta_in, weighted_sum):
        pass

    @abc.abstractmethod
    def get_number_of_parameters(self):
        pass

    @staticmethod
    def _get_stride(x, kernel_shape, stride):
        s0, s1 = x.strides[:2]
        m1, n1 = x.shape[:2]
        m2, n2 = kernel_shape[:2]
        view_shape = (1 + (m1 - m2) // stride, 1 + (n1 - n2) // stride, m2, n2) + x.shape[2:]
        strides = (stride * s0, stride * s1, s0, s1) + x.strides[2:]
        x_stride = np.lib.stride_tricks.as_strided(
            x, view_shape, strides=strides, writeable=False)
        return x_stride


class LinearLayer(BaseLayer):
    def __init__(self, input_shape, output_shape, activation_function=ActivationFunctionsForNN.Relu(),
                 lambda_regularization=0.01,
                 gradients_range=0.5, learning_rate=0.01):
        """
        create a linear layer
        :param input_shape: input size
        :param output_shape: output size
        :param activation_function: object of activation function, must be an instance of
        ActivationFunctionsForNN.BaseActivationFunctionForNN
        :param lambda_regularization: lambda regularization value
        :param gradients_range: range of gradients
        :param learning_rate: learning rate
        """
        if not isinstance(input_shape, int) or input_shape <= 0:
            raise InvalidArgumentException("input_shape must be a positive integer")
        if not isinstance(output_shape, int) or output_shape <= 0:
            raise InvalidArgumentException("output_shape must be a positive integer")
        if not isinstance(gradients_range, Number) or gradients_range < 0:
            raise InvalidArgumentException("gradients_range must be a non negative integer")
        if not isinstance(learning_rate, Number) or learning_rate <= 0:
            raise InvalidArgumentException("learning_rate must be a positive integer")
        if not isinstance(activation_function, ActivationFunctionsForNN.BaseActivationFunctionForNN):
            raise InvalidArgumentException("activation_function must inherit from BaseActivationFunctionForNN")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.lambda_regularization = lambda_regularization
        self.gradients_range = gradients_range

        self.weights = np.random.normal(0, np.sqrt(2 / self.input_shape), size=[self.output_shape, self.input_shape])
        self.biases = np.random.normal(0, np.sqrt(2 / self.input_shape), size=self.output_shape)

    def forward(self, x):
        results = np.dot(self.weights, x) + self.biases
        activated_results = self.activation_function.forward(results)
        return results, activated_results

    def back_propagate(self, delta_in, weighted_sum):
        return np.dot(self.weights.T, delta_in) * self.activation_function.backward(weighted_sum)

    def compute_gradients(self, delta, activation):
        gradients = np.outer(delta, activation)
        gradients = np.clip(gradients, -self.gradients_range, self.gradients_range)
        gradients_bias = np.sum(delta, axis=-1)

        return gradients, gradients_bias

    def gradient_decent(self, gradients, gradients_bias, number_of_records):
        self.weights = (self.weights * (1 - self.learning_rate * self.lambda_regularization / number_of_records)
                        - self.learning_rate * gradients / number_of_records)
        self.biases = self.biases - self.learning_rate * gradients_bias / number_of_records

    def get_number_of_parameters(self):
        return self.input_shape * self.output_shape + self.output_shape


class ConvolutionalLayer(BaseLayer):
    def __init__(self, kernel_size, input_channels, output_channels,
                 activation_function=ActivationFunctionsForNN.Relu(), lambda_regularization=0.01,
                 gradients_range=0.5, padding=0, stride=1, learning_rate=0.01):
        """
        create a convolutional layer
        :param kernel_size: size of the kernel
        :param input_channels: number of channels in input
        :param output_channels: number of channels in output
        :param activation_function: object of activation function, must be an instance of
        ActivationFunctionsForNN.BaseActivationFunctionForNN
        :param lambda_regularization: lambda regularization value
        :param padding: size of padding to add to each input
        :param stride: stride
        :param gradients_range: range of gradients
        :param learning_rate: learning rate
        """

        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise InvalidArgumentException("kernel_size must be a positive integer")
        if not isinstance(input_channels, int) or input_channels <= 0:
            raise InvalidArgumentException("input_shape must be a positive integer")
        if not isinstance(output_channels, int) or output_channels <= 0:
            raise InvalidArgumentException("output_shape must be a positive integer")
        if not isinstance(gradients_range, Number) or gradients_range < 0:
            raise InvalidArgumentException("gradients_range must be a non negative integer")
        if not isinstance(padding, int) or padding < 0:
            raise InvalidArgumentException("padding must be a non negative integer")
        if not isinstance(stride, int) or stride < 0:
            raise InvalidArgumentException("stride must be a non negative integer")
        if not isinstance(learning_rate, Number) or learning_rate <= 0:
            raise InvalidArgumentException("learning_rate must be a positive integer")
        if not isinstance(activation_function, ActivationFunctionsForNN.BaseActivationFunctionForNN):
            raise InvalidArgumentException("activation_function must inherit from BaseActivationFunctionForNN")

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.lambda_regularization = lambda_regularization
        self.gradients_range = gradients_range

        self.filters = np.array([np.random.normal(0, np.sqrt(2 / self.kernel_size ** 2 / self.input_channels),
                                                  [self.kernel_size, self.kernel_size, self.input_channels]) for _
                                 in range(self.output_channels)])

        self.biases = np.random.normal(0, np.sqrt(2 / self.kernel_size ** 2 / self.input_channels),
                                       self.output_channels)

    def forward(self, x):
        x = np.atleast_3d(x)
        x = self._add_padding(x, self.padding)

        weigh_sums = np.zeros([(x.shape[0] + 2 * self.padding - self.kernel_size) // self.stride + 1,
                               (x.shape[1] + 2 * self.padding - self.kernel_size) // self.stride + 1,
                               self.output_channels])

        for channel in range(self.output_channels):
            weigh_sums[:, :, channel] = self._conv3d(x, self.filters[channel])[:, :]

        results = weigh_sums + self.biases
        activated_results = self.activation_function.forward(results)
        return results, activated_results

    def back_propagate(self, delta_in, weighted_sum):
        result = np.zeros_like(weighted_sum)
        for output_channel in range(delta_in.shape[-1]):
            flipped_kernel = self.filters[output_channel, ::-1, ::-1, ...]
            for jj in range(weighted_sum.shape[-1]):
                result[:, :, jj] += self._full_conv3d(delta_in[:, :, output_channel], flipped_kernel[:, :, jj])
        result = result * self.activation_function.backward(weighted_sum)
        return result

    def compute_gradients(self, delta, activation):
        gradients = np.zeros_like(self.filters)
        for output_channel in range(delta.shape[-1]):
            delta_output_channel = np.take(delta, output_channel, axis=-1)
            gradiant_output_channel = gradients[output_channel]
            for activation_channel in range(activation.shape[-1]):
                gradiant_output_channel[:, :, activation_channel] += self._conv3d(activation[:, :, activation_channel],
                                                                                  delta_output_channel)
            gradients[output_channel] = np.clip(gradiant_output_channel, -self.gradients_range, self.gradients_range)
        gradients_bias = np.sum(delta, axis=(0, 1))
        return gradients, gradients_bias

    def gradient_decent(self, gradients, gradients_bias, number_of_records):
        self.filters = (self.filters * (1 - self.learning_rate * self.lambda_regularization / number_of_records)
                        - self.learning_rate * gradients / number_of_records)
        self.biases = self.biases - self.learning_rate * gradients_bias / number_of_records

    def get_number_of_parameters(self):
        return self.filters.size + self.output_channels

    @staticmethod
    def _add_padding(x, padding_left, padding_right=None):
        if padding_right is None:
            padding_right = padding_left
        if padding_left + padding_right == 0:
            return x
        x_padded = np.zeros(tuple(padding_left + padding_right + np.array(x.shape[:2])) + x.shape[2:])
        x_padded[padding_left:-padding_right, padding_left:-padding_right] = x
        return x_padded

    def _conv3d(self, x, kernel):
        if np.ndim(x) not in [2, 3]:
            raise InvalidArgumentException("Input must be in 2 or 3 dimensional")
        if np.ndim(kernel) not in [2, 3]:
            raise InvalidArgumentException("kernel must be 2 or 3 dimensional")
        if np.ndim(x) < np.ndim(kernel):
            raise InvalidArgumentException(
                "dimensions of the input must be more or equal than dimensions of the kernel")
        if np.ndim(x) == 3 and np.ndim(kernel) == 2:
            kernel = np.repeat(kernel[:, :, None], x.shape[2], axis=2)

        view = self._get_stride(self._add_padding(x, self.padding), kernel.shape, self.stride)
        return np.sum(view * kernel, axis=(2, 3)) if np.ndim(kernel) == 2 else np.sum(view * kernel, axis=(2, 3, 4))

    def _full_conv3d(self, x, kernel):
        shape = (x.shape[:2][0] + (self.stride - 1) * (x.shape[:2][0] - 1),
                 x.shape[:2][1] + (self.stride - 1) * (x.shape[:2][1] - 1)) + x.shape[2:]
        x_interlaced = np.zeros(shape)
        x_interlaced[0::self.stride, 0::self.stride, ...] = x
        pad_left = kernel.shape[:2][0] - 1
        idx = 0
        while True:
            idx_next = idx + self.stride
            win_left = idx_next - kernel.shape[:2][0] + 1
            if win_left <= x.shape[:2][0] - 1:
                idx = idx + self.stride
            else:
                break
        pad_right = idx - x.shape[:2][0] + 1
        x_padded = self._add_padding(x_interlaced, pad_left, pad_right)
        return self._conv3d(x_padded, kernel)


class PoolingLayer(BaseLayer):
    def __init__(self, add_padding=False, stride=1, use_mean_pooling=False):
        """
        create a pooling layer
        :param add_padding: whether the model should add padding to its inputs or not, if true, padding will be
        calculated automatically to keep the input size
        :param stride: stride
        :param use_mean_pooling: if true, will use mean pooling, else it will use max pooling
        """
        if not isinstance(stride, int) or stride < 0:
            raise InvalidArgumentException("stride must be a non negative integer")

        self.add_padding = add_padding
        self.stride = stride
        self.use_mean_pooling = use_mean_pooling
        if not self.use_mean_pooling:
            self.max_position = None

    def forward(self, x):
        x = np.atleast_3d(x)
        m, n = x.shape[:2]

        if self.add_padding:
            ny = m // self.stride + 1
            nx = n // self.stride + 1
            size = ((ny - 1) * self.stride + self.stride, (nx - 1) * self.stride + self.stride) + x.shape[2:]
            mat_pad = np.full(size, 0)
            mat_pad[:m, :n, ...] = x
        else:
            mat_pad = x[
                      :(m - self.stride) // self.stride * self.stride + self.stride,
                      :(n - self.stride) // self.stride * self.stride + self.stride, ...]

        view = self._get_stride(mat_pad, (self.stride, self.stride), self.stride)

        if self.use_mean_pooling:
            result = np.nanmean(view, axis=(2, 3))
        else:
            result = np.nanmax(view, axis=(2, 3), keepdims=True)
            self.max_position = np.where(result == view, 1, 0)
            result = np.squeeze(result)
        return result, result

    def back_propagate(self, delta_in, weighted_sum):
        result = np.zeros(weighted_sum.shape)
        if self.use_mean_pooling:
            sub_result = np.reshape(delta_in, (weighted_sum.shape[:2][0] // self.stride, 1,
                                               weighted_sum.shape[:2][1] // self.stride, 1)
                                    + delta_in.shape[2:])
            sub_result = np.repeat(sub_result, self.stride, axis=1)
            sub_result = np.repeat(sub_result, self.stride, axis=3)
            sub_result = np.reshape(sub_result, (weighted_sum.shape[:2][0] // self.stride * self.stride,
                                                 weighted_sum.shape[:2][1] // self.stride * self.stride)
                                    + delta_in.shape[2:])

            result[:sub_result.shape[0], :sub_result.shape[1], ...] = sub_result
        else:
            if not np.ndim(self.max_position) in [4, 5]:
                raise InvalidArgumentException(
                    f"Max position of a pooling layer must have 4 or 5 dimensions, not {np.ndim(self.max_position)}")

            if np.ndim(self.max_position) == 5:
                iy, ix, cy, cx, cc = np.where(self.max_position == 1)
                iy2 = iy * self.stride
                ix2 = ix * self.stride
                iy2 = iy2 + cy
                ix2 = ix2 + cx
                values = delta_in[iy, ix, cc].flatten()
                result[iy2, ix2, cc] = values
            else:
                iy, ix, cy, cx = np.where(self.max_position == 1)
                iy2 = iy * self.stride
                ix2 = ix * self.stride
                iy2 = iy2 + cy
                ix2 = ix2 + cx
                values = delta_in[iy, ix].flatten()
                result[iy2, ix2] = values

        return result

    def get_number_of_parameters(self):
        return 0


class FlattenLayer(BaseLayer):
    def __init__(self):
        """
        creates a flatting layer
        this layer flattens an N-dimensional input to a single dimensional array (similar to np.flatten)
        """
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        x = x.flatten()
        return x, x

    def back_propagate(self, delta_in, weighted_sum):
        return np.reshape(delta_in, self.input_shape)

    def get_number_of_parameters(self):
        return 0
