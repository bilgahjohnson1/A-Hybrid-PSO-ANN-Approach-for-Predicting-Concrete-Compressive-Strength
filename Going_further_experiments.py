import numpy as np
import pandas as pd
from train_ann import DataLoader, ANNTrainer
from ann import ANN
from pso import PSO
import json
import time
import matplotlib.pyplot as plt


class ANNBaselineTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, ann_architecture, activation_functions):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ann = ANN(ann_architecture, activation_functions)

    def evaluate(self):
        preds_train = self.ann.predict(self.X_train).flatten()
        preds_test = self.ann.predict(self.X_test).flatten()
        
        train_mae = np.mean(np.abs(preds_train - self.y_train))
        test_mae = np.mean(np.abs(preds_test - self.y_test))
        
        return train_mae, test_mae


#CLERC & KENNEDY CONSTRICTION FACTOR (χ)
def compute_constriction_factor(c1, c2):
    phi = c1 + c2
    if phi <= 4:
        phi = 4.00001
    chi = 2 / abs(2 - phi - np.sqrt(phi**2 - 4*phi))
    return chi


class PSO_WithConstriction(PSO):
    def __init__(self, *args, use_constriction=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_constriction = use_constriction
        
    def optimize(self, fitness_function, verbose=True):
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                f = fitness_function(self.positions[i])
                self.fitnesses[i] = f

                if f < self.personal_best_fitnesses[i]:
                    self.personal_best_fitnesses[i] = f
                    self.personal_best_positions[i] = self.positions[i].copy()

                if f < self.global_best_fitness:
                    self.global_best_fitness = f
                    self.global_best_position = self.positions[i].copy()

            if self.use_constriction:
                chi = compute_constriction_factor(self.beta, self.gamma)
            else:
                chi = 1.0

            for i in range(self.num_particles):
                p_best = self.personal_best_positions[i]
                informant_best = self._informant_best(i)
                global_best = self.global_best_position

                for d in range(self.num_dimensions):
                    b = np.random.uniform(0, self.beta)
                    c = np.random.uniform(0, self.gamma)
                    d_rand = np.random.uniform(0, self.delta)

                    self.velocities[i, d] = chi * (
                        self.alpha * self.velocities[i, d] +
                        b * (p_best[d] - self.positions[i, d]) +
                        c * (informant_best[d] - self.positions[i, d]) +
                        d_rand * (global_best[d] - self.positions[i, d])
                    )

            self.positions += self.e * self.velocities
            self.positions = self._apply_bounds(self.positions)
            self.fitness_history.append(self.global_best_fitness)

            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"Iter {iteration:3d}: Best = {self.global_best_fitness:.6f}")

        return self.global_best_position, self.global_best_fitness


# CELL 3: MULTIPLE BOUNDARY HANDLING STRATEGIES
class PSO_WithBoundaryHandling(PSO):
    def __init__(self, *args, boundary_method="clip", **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_method = boundary_method
        
    def _apply_bounds(self, x):
        min_b, max_b = self.bounds
        
        if self.boundary_method == "clip":
            return np.clip(x, min_b, max_b)
        
        elif self.boundary_method == "reflect":
            x = np.where(x < min_b, min_b + (min_b - x), x)
            x = np.where(x > max_b, max_b - (x - max_b), x)
            return np.clip(x, min_b, max_b)
        
        elif self.boundary_method == "random":
            return np.where((x < min_b) | (x > max_b),
                          np.random.uniform(min_b, max_b, size=x.shape), x)
        
        elif self.boundary_method == "absorb":
            outside_mask = (x < min_b) | (x > max_b)
            if np.any(outside_mask):
                self.velocities[outside_mask] = 0
            return np.clip(x, min_b, max_b)
        
        else:
            return np.clip(x, min_b, max_b)


# CELL 4: EARLY STOPPING LOGIC FOR PSO
def should_stop_early(history, patience=30, min_delta=1e-5):
    if len(history) < patience:
        return False
    recent = history[-patience:]
    improvement = max(recent) - min(recent)
    return improvement < min_delta


class PSO_WithEarlyStopping(PSO):
    def __init__(self, *args, early_stop_patience=30, early_stop_delta=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        
    def optimize(self, fitness_function, verbose=True):
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                f = fitness_function(self.positions[i])
                self.fitnesses[i] = f

                if f < self.personal_best_fitnesses[i]:
                    self.personal_best_fitnesses[i] = f
                    self.personal_best_positions[i] = self.positions[i].copy()

                if f < self.global_best_fitness:
                    self.global_best_fitness = f
                    self.global_best_position = self.positions[i].copy()

            for i in range(self.num_particles):
                p_best = self.personal_best_positions[i]
                informant_best = self._informant_best(i)
                global_best = self.global_best_position

                for d in range(self.num_dimensions):
                    b = np.random.uniform(0, self.beta)
                    c = np.random.uniform(0, self.gamma)
                    d_rand = np.random.uniform(0, self.delta)

                    self.velocities[i, d] = (
                        self.alpha * self.velocities[i, d] +
                        b * (p_best[d] - self.positions[i, d]) +
                        c * (informant_best[d] - self.positions[i, d]) +
                        d_rand * (global_best[d] - self.positions[i, d])
                    )

            self.positions += self.e * self.velocities
            self.positions = self._apply_bounds(self.positions)
            self.fitness_history.append(self.global_best_fitness)

            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"Iter {iteration:3d}: Best = {self.global_best_fitness:.6f}")

            if should_stop_early(self.fitness_history, self.early_stop_patience, self.early_stop_delta):
                print(f"Early stopping triggered at iteration {iteration}")
                break

        return self.global_best_position, self.global_best_fitness


# CELL 5: PLOT PSO CONVERGENCE CURVE
def plot_convergence(history, save_path="convergence_curve.png", title="PSO Convergence Curve"):
    plt.figure(figsize=(10, 5))
    plt.plot(history, label="Best Fitness (MAE)", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Fitness (MAE)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# CELL 6: CUSTOM TRAINER WITH EXTENDED PSO OPTIONS
class ExtendedANNTrainer(ANNTrainer):
    def train(self, pso_params, pso_class=PSO):
        print("\n" + "=" * 60)
        print("TRAINING ANN WITH PSO")
        print("=" * 60)
        print(f"PSO Class: {pso_class.__name__}")
        print(f"Total ANN parameters to optimize: {self.num_parameters}")
        
        pso = pso_class(
            num_particles=pso_params.get('num_particles', 30),
            num_dimensions=self.num_parameters,
            num_informants=pso_params.get('num_informants', 3),
            max_iterations=pso_params.get('max_iterations', 100),
            bounds=pso_params.get('bounds', (-2.0, 2.0)),
            alpha=pso_params.get('alpha', 0.7),
            beta=pso_params.get('beta', 1.0),
            gamma=pso_params.get('gamma', 1.0),
            delta=pso_params.get('delta', 0.0),
            step_size=pso_params.get('step_size', 1.0),
            **{k: v for k, v in pso_params.items() 
               if k not in ['num_particles', 'num_dimensions', 'num_informants', 
                           'max_iterations', 'bounds', 'alpha', 'beta', 'gamma', 
                           'delta', 'step_size']}
        )
        
        best_params, best_fitness = pso.optimize(self.fitness_function, verbose=True)
        self.ann.set_parameters(best_params)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Best fitness achieved: {best_fitness:.6f}")
        print("=" * 60)
        
        return best_params, best_fitness, pso.fitness_history


#MAIN EXPERIMENT
def run_going_further_experiments():
    print("=" * 70)
    print("GOING FURTHER EXPERIMENTS")
    print("=" * 70)
    
    # Load data
    print("\nLoading concrete dataset...")
    data_loader = DataLoader('concrete_Data.csv')
    X_train, y_train, X_test, y_test = data_loader.train_test_split(train_ratio=0.7)
    
    ann_architecture = [8, 10, 1]
    activation_functions = ['relu', 'linear']
    results = {}
    
    # 1. BASELINE: ANN WITHOUT PSO
    print("\n" + "=" * 70)
    print("[1] BASELINE: ANN WITHOUT PSO OPTIMISATION")
    print("=" * 70)
    baseline = ANNBaselineTrainer(X_train, y_train, X_test, y_test,
                                  ann_architecture, activation_functions)
    b_train, b_test = baseline.evaluate()
    print(f"Baseline MAE - Training: {b_train:.6f}, Testing: {b_test:.6f}")
    results['baseline'] = {'train_mae': b_train, 'test_mae': b_test}
    
    # Standard PSO for comparison
    print("\n" + "=" * 70)
    print("[1b] STANDARD PSO (for comparison)")
    print("=" * 70)
    trainer_standard = ExtendedANNTrainer(X_train, y_train, X_test, y_test,
                                         ann_architecture, activation_functions)
    pso_params_standard = {
        'num_particles': 30,
        'max_iterations': 100,
        'num_informants': 3,
        'bounds': (-2, 2),
        'alpha': 0.7,
        'beta': 1.5,
        'gamma': 1.5,
        'delta': 0.0,
        'step_size': 0.5
    }
    _, _, history_standard = trainer_standard.train(pso_params_standard, PSO)
    train_mae_std, _ = trainer_standard.evaluate(on_test=False)
    test_mae_std, _ = trainer_standard.evaluate(on_test=True)
    plot_convergence(history_standard, "conv_standard.png", "Standard PSO Convergence")
    results['standard_pso'] = {'train_mae': train_mae_std, 'test_mae': test_mae_std}
    
    # 2. PSO WITH CONSTRICTION FACTOR
    print("\n" + "=" * 70)
    print("[2] PSO WITH CONSTRICTION FACTOR (χ)")
    print("=" * 70)
    trainer_constriction = ExtendedANNTrainer(X_train, y_train, X_test, y_test,
                                             ann_architecture, activation_functions)
    pso_params_constriction = pso_params_standard.copy()
    pso_params_constriction['use_constriction'] = True
    
    _, _, history_constriction = trainer_constriction.train(pso_params_constriction, 
                                                            PSO_WithConstriction)
    train_mae_const, _ = trainer_constriction.evaluate(on_test=False)
    test_mae_const, _ = trainer_constriction.evaluate(on_test=True)
    plot_convergence(history_constriction, "conv_constriction.png", 
                    "PSO with Constriction Factor")
    results['constriction'] = {'train_mae': train_mae_const, 'test_mae': test_mae_const}
    
    # 3. BOUNDARY HANDLING METHODS
    print("\n" + "=" * 70)
    print("[3] BOUNDARY HANDLING METHODS")
    print("=" * 70)
    
    for method in ["clip", "reflect", "random", "absorb"]:
        print(f"\nTesting boundary method: {method}")
        trainer_boundary = ExtendedANNTrainer(X_train, y_train, X_test, y_test,
                                             ann_architecture, activation_functions)
        pso_params_boundary = pso_params_standard.copy()
        pso_params_boundary['boundary_method'] = method
        
        _, _, history_boundary = trainer_boundary.train(pso_params_boundary, 
                                                       PSO_WithBoundaryHandling)
        train_mae_bound, _ = trainer_boundary.evaluate(on_test=False)
        test_mae_bound, _ = trainer_boundary.evaluate(on_test=True)
        plot_convergence(history_boundary, f"conv_boundary_{method}.png", 
                        f"Boundary Method: {method.title()}")
        results[f'boundary_{method}'] = {'train_mae': train_mae_bound, 'test_mae': test_mae_bound}
    
    # 4. INFORMANT NETWORK SIZE STUDY
    print("\n" + "=" * 70)
    print("[4] INFORMANT NETWORK SIZE STUDY")
    print("=" * 70)
    
    for k in [1, 3, 5, 10]:
        print(f"\nTesting num_informants = {k}")
        trainer_informants = ExtendedANNTrainer(X_train, y_train, X_test, y_test,
                                               ann_architecture, activation_functions)
        pso_params_informants = pso_params_standard.copy()
        pso_params_informants['num_informants'] = k
        
        _, _, history_informants = trainer_informants.train(pso_params_informants, PSO)
        train_mae_inf, _ = trainer_informants.evaluate(on_test=False)
        test_mae_inf, _ = trainer_informants.evaluate(on_test=True)
        plot_convergence(history_informants, f"conv_informants_{k}.png", 
                        f"Informants: k={k}")
        results[f'informants_{k}'] = {'train_mae': train_mae_inf, 'test_mae': test_mae_inf}
    
    # 5. EARLY STOPPING
    print("\n" + "=" * 70)
    print("[5] EARLY STOPPING")
    print("=" * 70)
    trainer_early = ExtendedANNTrainer(X_train, y_train, X_test, y_test,
                                      ann_architecture, activation_functions)
    pso_params_early = pso_params_standard.copy()
    pso_params_early['early_stop_patience'] = 20
    pso_params_early['early_stop_delta'] = 1e-5
    
    _, _, history_early = trainer_early.train(pso_params_early, PSO_WithEarlyStopping)
    train_mae_early, _ = trainer_early.evaluate(on_test=False)
    test_mae_early, _ = trainer_early.evaluate(on_test=True)
    plot_convergence(history_early, "conv_early_stopping.png", 
                    "PSO with Early Stopping")
    results['early_stopping'] = {'train_mae': train_mae_early, 'test_mae': test_mae_early}
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Train MAE':<15} {'Test MAE':<15}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<30} {res['train_mae']:<15.6f} {res['test_mae']:<15.6f}")
    
    print("\n" + "=" * 70)
    print("GOING FURTHER EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    return results

# RUNNING EXPERIMENTS
if __name__ == "__main__":
    results = run_going_further_experiments()
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find best and worst performers
    best_exp = min(results.items(), key=lambda x: x[1]['test_mae'])
    worst_exp = max(results.items(), key=lambda x: x[1]['test_mae'])
    
    print(f"Best Configuration: {best_exp[0]}")
    print(f"Test MAE: {best_exp[1]['test_mae']:.6f}")
    
    print(f"Worst Configuration: {worst_exp[0]}")
    print(f"Test MAE: {worst_exp[1]['test_mae']:.6f}")
    
    improvement = ((worst_exp[1]['test_mae'] - best_exp[1]['test_mae']) / 
                   worst_exp[1]['test_mae'] * 100)
    print(f"Improvement from worst to best: {improvement:.2f}%")
    
    # Compare baseline vs PSO
    if 'baseline' in results and 'standard_pso' in results:
        baseline_mae = results['baseline']['test_mae']
        pso_mae = results['standard_pso']['test_mae']
        pso_improvement = ((baseline_mae - pso_mae) / baseline_mae * 100)
        print(f"PSO Improvement over Random Weights: {pso_improvement:.2f}%")
        print(f"Baseline (no training): {baseline_mae:.6f}")
        print(f"With PSO training: {pso_mae:.6f}")
    
