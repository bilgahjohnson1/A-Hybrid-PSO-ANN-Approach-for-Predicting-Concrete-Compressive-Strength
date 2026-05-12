import numpy as np
import pandas as pd
from ann import ANN
from pso import PSO

class DataLoader:  
    def __init__(self, filename='CODE/concrete_data.csv'):
        # Loading the CSV file
        self.data = pd.read_csv(filename)
        
        print(f"\nLoaded dataset: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
        print(f"Columns: {list(self.data.columns)}")
        
        # Separate features (X) and target (y)
        # First 8 columns are features, last column is target
        self.X = self.data.iloc[:, :-1].values  # All columns except last
        self.y = self.data.iloc[:, -1].values   # Last column only
        
        # Normalize features to [0, 1] range for better ANN performance
        # Store min/max for each feature
        self.X_min = self.X.min(axis=0)
        self.X_max = self.X.max(axis=0)
        self.X_normalized = (self.X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        # Normalize target as well
        self.y_min = self.y.min()
        self.y_max = self.y.max()
        self.y_normalized = (self.y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        
        print(f"\nData normalized")
        print(f"Feature range: [0, 1]")
        print(f"Target range: [0, 1] (original: [{self.y_min:.2f}, {self.y_max:.2f}])")
    
    #Splitting the data into taining and testing sets
    def train_test_split(self, train_ratio=0.7, random_seed=42):
        np.random.seed(random_seed)
        
        # Shuffling the indices
        indices = np.arange(len(self.X_normalized))
        np.random.shuffle(indices)
        
        # Calculating split point
        split_idx = int(len(indices) * train_ratio)
        
        # Spliting the indices
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Spliting the data
        X_train = self.X_normalized[train_indices]
        y_train = self.y_normalized[train_indices]
        X_test = self.X_normalized[test_indices]
        y_test = self.y_normalized[test_indices]
        
        print(f"\nData split: {len(X_train)} training, {len(X_test)} testing samples")
        
        return X_train, y_train, X_test, y_test
    
    def denormalize_output(self, y_normalized):
        return y_normalized * (self.y_max - self.y_min) + self.y_min

#Coupling PSO with ANN to train the network
class ANNTrainer:

    #Initializing the trainer
    def __init__(self, X_train, y_train, X_test, y_test, ann_architecture, activation_functions):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Creating the ANN
        self.ann = ANN(ann_architecture, activation_functions)
        
        # Get number of parameters needed
        self.num_parameters = self.ann.get_parameter_count()
        
        print(f"\nANN created: {ann_architecture}")
        print(f"Activations: {activation_functions}")
        print(f"Total parameters to optimize: {self.num_parameters}")
    
    #Fitness function to minimize PSO
    def fitness_function(self, parameters):

        # Seting ANN parameters from PSO particle
        self.ann.set_parameters(parameters)
        
        # Making predictions on training data
        predictions = self.ann.predict(self.X_train)
        
        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(predictions.flatten() - self.y_train))
        
        return mae
    
    #Evaluating current ANN performance
    def evaluate(self, on_test=True):
        if on_test:
            X = self.X_test
            y = self.y_test
            dataset_name = "test"
        else:
            X = self.X_train
            y = self.y_train
            dataset_name = "training"
        
        predictions = self.ann.predict(X)
        mae = np.mean(np.abs(predictions.flatten() - y))
        rmse = np.sqrt(np.mean((predictions.flatten() - y) ** 2))

        
        print(f"\nMAE on {dataset_name} set: {mae:.6f}")
        print(f"RMSE on {dataset_name} set: {rmse:.6f}")

        return mae, rmse
        
    #Training the ANN using PSO
    def train(self, pso_params):
        print("\nTRAINING ANN WITH PSO")
        print(f"Parameters: {pso_params}")
        print(f"Total ANN parameters to optimize: {self.num_parameters}")
        print("Starting optimization")
        
        # PSO optimizer
        pso = PSO(
            num_particles=pso_params.get('num_particles', 30),
            num_dimensions=self.num_parameters,
            num_informants=pso_params.get('num_informants', 3),
            max_iterations=pso_params.get('max_iterations', 100),
            bounds=pso_params.get('bounds', (-8.0, 8.0)),
            alpha=pso_params.get('alpha', 0.7),
            beta=pso_params.get('beta', 1.49445),
            gamma=pso_params.get('gamma', 1.49445),
            delta=pso_params.get('delta',0.0),
            step_size=pso_params.get('step_size',1.0)

        )
        
        # Run PSO optimization
        best_params, best_fitness = pso.optimize(
            self.fitness_function,
            verbose=True
        )
        
        # Seting ANN to best parameters found
        self.ann.set_parameters(best_params)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Best fitness achieved: {best_fitness:.6f}")
        print("=" * 60)
        
        return best_params, best_fitness

# TESTING CODE FOR PART 3
if __name__ == "__main__":
    print("PSO-ANN INTEGRATION TEST")
    print("=" * 60)
    
    # Step 1: Loading the data
    print("\nLoading the dataset")
    data_loader = DataLoader('CODE/concrete_data.csv')
    
    # Step 2: Spliting data
    print("\nSplitting data")
    X_train, y_train, X_test, y_test = data_loader.train_test_split(train_ratio=0.7)
    
    # Step 3: Creating ANN architecture
    print("\nCreating ANN")
    ann_architecture = [8, 10, 1]  # 8 inputs, 10 hidden neurons, 1 output
    activation_functions = ['relu', 'linear']  # ReLU for hidden, linear for output
    
    # Step 4: Creating trainer
    trainer = ANNTrainer(
        X_train, y_train,
        X_test, y_test,
        ann_architecture,
        activation_functions
    )
    
    # Step 5: Test with random weights
    print("\nTesting with random weights")
    print("Before training:")
    trainer.evaluate(on_test=False)
    trainer.evaluate(on_test=True)
    
    # Step 6: Training with PSO
    print("\nTraining with PSO")
    pso_params = {
        'num_particles': 50,      
        'max_iterations': 300,     
        'num_informants': 3,
        'bounds': (-2.0, 2.0)
    }
    
    print(f"\nNote: Using {pso_params['num_particles']} particles × {pso_params['max_iterations']} iterations")
    print(f"Total evaluations: {pso_params['num_particles'] * pso_params['max_iterations']}")

    best_params, best_fitness = trainer.train(pso_params)

    print(f"\nBest fitness from PSO: {best_fitness:.6f}")
    
    # Step 7: Evaluating the trained ANN
    print("\nEvaluating trained ANN")
    print("After training:")
    train_mae, train_rmse = trainer.evaluate(on_test=False)
    test_mae, test_rmse = trainer.evaluate(on_test=True)
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Training MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
    print(f"  Testing MAE:  {test_mae:.6f}, RMSE: {test_rmse:.6f}")
    print(f"\nThese are normalized values (0-1 range)")
    
    # REAL-WORLD (MPa)
    train_predictions_denorm = data_loader.denormalize_output(trainer.ann.predict(trainer.X_train).flatten())
    train_actual_denorm = data_loader.denormalize_output(trainer.y_train)
    test_predictions_denorm = data_loader.denormalize_output(trainer.ann.predict(trainer.X_test).flatten())
    test_actual_denorm = data_loader.denormalize_output(trainer.y_test)

    train_mae_denorm = np.mean(np.abs(train_predictions_denorm - train_actual_denorm))
    train_rmse_denorm = np.sqrt(np.mean((train_predictions_denorm - train_actual_denorm) ** 2))
    test_mae_denorm = np.mean(np.abs(test_predictions_denorm - test_actual_denorm))
    test_rmse_denorm = np.sqrt(np.mean((test_predictions_denorm - test_actual_denorm) ** 2))

    print("\nREAL-WORLD (MPa)")
    print("==============================")
    print(f"  Training MAE: {train_mae_denorm:.2f} MPa, RMSE: {train_rmse_denorm:.2f} MPa")
    print(f"  Testing MAE:  {test_mae_denorm:.2f} MPa, RMSE: {test_rmse_denorm:.2f} MPa")

    # ACCURACY (REGRESSION APPROX)
    target_range = data_loader.y_max - data_loader.y_min

    train_accuracy = 1 - (train_mae_denorm / target_range)
    test_accuracy = 1 - (test_mae_denorm / target_range)

    print("\nACCURACY (based on MAE / target range)")
    print("==============================")
    print(f"  Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"  Testing Accuracy:  {test_accuracy * 100:.2f}%")