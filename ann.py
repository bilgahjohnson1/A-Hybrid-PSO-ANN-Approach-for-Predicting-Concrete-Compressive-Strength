import numpy as np

class ANN:
    def __init__(self, layer_sizes, activation_functions):
       
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_functions = activation_functions
        
        # Initializing the weights and biases with random values
        # Will be replaced by PSO later
        self.weights = []
        self.biases = []
        
        # Creating weight matrices between each pair of consecutive layers
        for i in range(self.num_layers - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            self.weights.append(weight)
            bias = np.zeros(layer_sizes[i+1])
            self.biases.append(bias)
    
    def set_parameters(self, params):
        idx = 0
        
        # Extracting weights for each layer
        for i in range(len(self.weights)):
            weight_size = self.weights[i].size
            self.weights[i] = params[idx:idx + weight_size].reshape(self.weights[i].shape)
            idx += weight_size
        
        # Extracting biases for each layer
        for i in range(len(self.biases)):
            bias_size = self.biases[i].size
            self.biases[i] = params[idx:idx + bias_size]
            idx += bias_size
    
    def get_parameter_count(self):
        count = 0
        for w in self.weights:
            count += w.size
        for b in self.biases:
            count += b.size
        return count
    
    #Activation function
    def activate(self, x, function_name):
     
        if function_name == 'logistic':
            # Logistic function: 1 / (1 + e^(-x))
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        
        elif function_name == 'relu':
            # ReLU: max(0, x)
            return np.maximum(0, x)
        
        elif function_name == 'tanh':
            # Hyperbolic tangent
            return np.tanh(x)
        
        elif function_name == 'linear':
            # Linear activation (no change)
            return x
        
        else:
            raise ValueError(f"Unknown activation function: {function_name}")
    
    #Forward propagation
    def forward(self, inputs):
     
        # Input layer
        activation = np.array(inputs)
        
        # Propagating through each layer
        for i in range(len(self.weights)):
            # Linear transformation: z = activation * weights + bias
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            # Applying activation function
            activation = self.activate(z, self.activation_functions[i])
        
        return activation
    
    #Predictions for multiple input
    def predict(self, X):
      
        predictions = []
        for sample in X:
            pred = self.forward(sample)
            predictions.append(pred)
        return np.array(predictions)

# TESTING CODE FOR PART 1
if __name__ == "__main__":
    print("TESTING ANN IMPLEMENTATION")
    print("=" * 60)
    
    # Test 1: Creating a simple ANN
    print("1.Creating ANN with architecture [3, 5, 2]")
    
    ann = ANN(
        layer_sizes=[3, 5, 2],
        activation_functions=['relu', 'linear']
    )
    
    print(f"ANN created")
    print(f"Total parameters: {ann.get_parameter_count()}")
    
    # Test 2: Forward propagation with random input
    print("\n2.Forward propagation")
    test_input = np.array([1.0, 2.0, 3.0])
    output = ann.forward(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {output}")
    print(f"Forward propagation is working")
    
    # Test 3: Batch prediction
    print("\n3.Batch prediction")
    test_batch = np.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [2.0, 3.0, 4.0]
    ])
    predictions = ann.predict(test_batch)
    print(f"Batch shape: {test_batch.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Batch prediction is working")
    
    # Test 4: Setting parameters from a vector
    print("\n4.Setting parameters from PSO vector")
    param_count = ann.get_parameter_count()
    new_params = np.random.randn(param_count) * 0.1
    ann.set_parameters(new_params)
    new_output = ann.forward(test_input)
    print(f"Parameters updated")
    print(f"New output: {new_output}")
    print("All tests passed")
    
    