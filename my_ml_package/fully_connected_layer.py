class FullyConnectedLayer:
    """Fully connected layer class for neural network model"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.input_shape = None
        self.output_shape = None

    def forward(self, inputs):
        """Forward pass of the fully connected layer"""
        return inputs @ self.weights + self.bias
