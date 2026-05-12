import numpy as np
import pandas as pd
from train_ann import DataLoader, ANNTrainer
import json
import time

class ExperimentRunner:
    
    def __init__(self, data_loader, num_runs=10):
        self.data_loader = data_loader
        self.num_runs = num_runs
        
        # Split the data
        self.X_train, self.y_train, self.X_test, self.y_test = \
            data_loader.train_test_split(train_ratio=0.7)
        
        # Store results
        self.results = []
    
    def run_single_experiment(self, ann_architecture, activation_functions, pso_params, run_number):
        print(f"\n--- Run {run_number + 1}/{self.num_runs} ---")
        
        # Create trainer
        trainer = ANNTrainer(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            ann_architecture,
            activation_functions
        )
        
        # Train
        start_time = time.time()
        best_params, best_fitness = trainer.train(pso_params)
        training_time = time.time() - start_time
        
        # Evaluate on both sets
        train_mae = trainer.evaluate(on_test=False)
        test_mae = trainer.evaluate(on_test=True)
        
        return {
            'run': run_number,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'best_fitness': best_fitness,
            'training_time': training_time
        }
    
    #Running multiple runs of the same configuration
    def run_experiment_set(self, experiment_name, ann_architecture, activation_functions, pso_params):

        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {experiment_name}")
        print("=" * 70)
        print(f"ANN Architecture: {ann_architecture}")
        print(f"Activations: {activation_functions}")
        print(f"PSO Params: {pso_params}")
        print(f"Running {self.num_runs} independent runs...")
        
        run_results = []
        
        # Run multiple times
        for run in range(self.num_runs):
            result = self.run_single_experiment(
                ann_architecture,
                activation_functions,
                pso_params,
                run
            )
            run_results.append(result)
        
        # Calculating statistics
        train_maes = [r['train_mae'] for r in run_results]
        test_maes = [r['test_mae'] for r in run_results]
        times = [r['training_time'] for r in run_results]
        
        summary = {
            'experiment_name': experiment_name,
            'ann_architecture': ann_architecture,
            'activation_functions': activation_functions,
            'pso_params': pso_params,
            'train_mae_mean': np.mean(train_maes),
            'train_mae_std': np.std(train_maes),
            'test_mae_mean': np.mean(test_maes),
            'test_mae_std': np.std(test_maes),
            'avg_time': np.mean(times),
            'individual_runs': run_results
        }
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {experiment_name}")
        print(f"{'='*70}")
        print(f"Training MAE: {summary['train_mae_mean']:.6f} ± {summary['train_mae_std']:.6f}")
        print(f"Testing MAE:  {summary['test_mae_mean']:.6f} ± {summary['test_mae_std']:.6f}")
        print(f"Avg Time:     {summary['avg_time']:.2f} seconds")
        
        self.results.append(summary)
        return summary
    
    #save all results to JSON file
    def save_results(self, filename='experiment_results.json'):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary_table(self):
        print("\n" + "=" * 100)
        print("SUMMARY OF ALL EXPERIMENTS")
        print("=" * 100)
        print(f"{'Experiment':<40} {'Activations':<25} {'Train MAE':<20} {'Test MAE':<20} {'Time (s)':<10}")
        print("-" * 100)
        
        for result in self.results:
            name = result['experiment_name'][:39]
            activs = ", ".join(result['activation_functions'])
            train_mae = f"{result['train_mae_mean']:.6f} ± {result['train_mae_std']:.6f}"
            test_mae = f"{result['test_mae_mean']:.6f} ± {result['test_mae_std']:.6f}"
            time_str = f"{result['avg_time']:.2f}"
            
            print(f"{name:<40} {activs:<25} {train_mae:<20} {test_mae:<20} {time_str:<10}")
        
        print("=" * 100)

    # def print_summary_table(self):
    #     """
    #     Print a summary table of all experiments.
    #     """
    #     print("\n" + "=" * 100)
    #     print("SUMMARY OF ALL EXPERIMENTS")
    #     print("=" * 100)
    #     print(f"{'Experiment':<40} {'Activations':<25} {'Train MAE':<20} {'Test MAE':<20} {'Time (s)':<10}")
    #     print("-" * 100)
        
    #     for result in self.results:
    #         name = result['experiment_name'][:39]
    #         activs = ", ".join(result['activation_functions'])
    #         train_mae = f"{result['train_mae_mean']:.6f} ± {result['train_mae_std']:.6f}"
    #         test_mae = f"{result['test_mae_mean']:.6f} ± {result['test_mae_std']:.6f}"
    #         time_str = f"{result['avg_time']:.2f}"
            
    #         print(f"{name:<40} {activs:<25} {train_mae:<20} {test_mae:<20} {time_str:<10}")
        
    #     print("=" * 100)

# QUESTION 1: ANN ARCHITECTURE EFFECT
def investigate_architecture(runner):
    print("\n" + "#" * 70)
    print("# INVESTIGATION 1: ANN ARCHITECTURE EFFECT")
    print("#" * 70)
    
    # Base PSO parameters
    base_pso = {
        'num_particles': 20,
        'max_iterations': 50,
        'num_informants': 3,
        'bounds': (-2.0, 2.0)
    }
    
    # Test 1: Single hidden layer with different sizes
    print("\n## Test 1.1: Different hidden layer sizes")
    for hidden_size in [5, 10, 15]:
        runner.run_experiment_set(
            experiment_name=f"Architecture: [8, {hidden_size}, 1]",
            ann_architecture=[8, hidden_size, 1],
            activation_functions=['relu', 'linear'],
            pso_params=base_pso
        )
    
    # Test 2: Multiple hidden layers
    print("\n## Test 1.2: Multiple hidden layers")
    runner.run_experiment_set(
        experiment_name="Architecture: [8, 10, 5, 1] (2 hidden layers)",
        ann_architecture=[8, 10, 5, 1],
        activation_functions=['relu', 'relu', 'linear'],
        pso_params=base_pso
    )
    
    # Test 3: Different activation functions
    print("\n## Test 1.3: Different activation functions")
    runner.run_experiment_set(
        experiment_name="Activation: Tanh in hidden layer",
        ann_architecture=[8, 10, 1],
        activation_functions=['tanh', 'linear'],
        pso_params=base_pso
    )


# QUESTION 2: SWARM SIZE VS ITERATIONS
def investigate_evaluation_budget(runner):
    print("# INVESTIGATION 2: EVALUATION BUDGET ALLOCATION")
    print("=" * 100)
    print("Fixed budget: 500 evaluations")
    print("Testing different (swarm_size, iterations) combinations")
    
    # Base architecture- kept constant
    architecture = [8, 10, 1]
    activations = ['relu', 'linear']
    
    # Different allocations that all give 500 evaluations
    allocations = [
        (10, 50),   # Small swarm, many iterations
        (20, 25),   # Balanced
        (25, 20),   # Balanced (reversed)
        (50, 10),   # Large swarm, few iterations
    ]
    
    for swarm_size, iterations in allocations:
        pso_params = {
            'num_particles': swarm_size,
            'max_iterations': iterations,
            'num_informants': 3,
            'bounds': (-2.0, 2.0)
        }
        
        runner.run_experiment_set(
            experiment_name=f"Budget: {swarm_size} particles × {iterations} iter",
            ann_architecture=architecture,
            activation_functions=activations,
            pso_params=pso_params
        )

# QUESTION 3: PSO ACCELERATION COEFFICIENTS
def investigate_acceleration_coefficients(runner):
    print("=" * 100)
    print("# INVESTIGATION 3: PSO ACCELERATION COEFFICIENTS")
    print("=" * 100)
    
    # Base setup
    architecture = [8, 10, 1]
    activations = ['relu', 'linear']
    base_pso = {
        'num_particles': 20,
        'max_iterations': 50,
        'num_informants': 3,
        'bounds': (-2.0, 2.0),
        'inertia_weight': 0.729
    }
    
    # Test different coefficient combinations
    coefficient_sets = [
        (2.0, 0.5, "High cognitive, low social"),
        (1.49, 1.49, "Balanced (standard)"),
        (0.5, 2.0, "Low cognitive, high social"),
        (2.5, 2.5, "Both high"),
    ]
    
    for c1, c2, description in coefficient_sets:
        pso_params = base_pso.copy()
        pso_params['cognitive_coef'] = c1
        pso_params['social_coef'] = c2
        
        runner.run_experiment_set(
            experiment_name=f"Coefficients: c1={c1}, c2={c2} ({description})",
            ann_architecture=architecture,
            activation_functions=activations,
            pso_params=pso_params
        )


# MAIN EXPERIMENTAL SCRIPT
if __name__ == "__main__":
    print("=" * 100)
    print("EXPERIMENTAL INVESTIGATION")
    print("=" * 100)
    
    # Load data
    data_loader = DataLoader('CODE/concrete_data.csv')
    
    # Create experiment runner
    runner = ExperimentRunner(data_loader, num_runs=10)
    
    # Run investigations
    print("=" * 100)
    print("Starting experiments")
    print("=" * 100)
    
    # Run each investigation
    investigate_architecture(runner)
    investigate_evaluation_budget(runner)
    investigate_acceleration_coefficients(runner)
    
    # Print summary
    runner.print_summary_table()
    
    # Save results
    runner.save_results('experiment_results.json')
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    
