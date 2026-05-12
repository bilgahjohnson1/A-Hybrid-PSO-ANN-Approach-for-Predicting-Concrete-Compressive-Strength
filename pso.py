import numpy as np

class PSO: #using algorithm 39
   
    def __init__(self,
                 num_particles,
                 num_dimensions,
                 num_informants=3,
                 max_iterations=100,
                 bounds=(-1.0, 1.0),
                 alpha=0.7,      # α: velocity retention
                 beta=1.0,       # β: personal best to be retained
                 gamma=1.0,      # γ: informant's best to be retained
                 delta=0.0,      # δ: global best to be retained
                 step_size=1.0   # e: jump size of a particle
                 ):

        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_informants = num_informants
        self.max_iterations = max_iterations
        self.bounds = bounds

        # Algorithm 39 parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.e = step_size

        # (Lines 1–6) Initializing particle positions
        self.positions = np.random.uniform(bounds[0], bounds[1],
                                           (num_particles, num_dimensions))

        # (Lines 7–9) Initializing particle velocities to zero
        self.velocities = np.zeros((num_particles, num_dimensions))

        # (Lines 10–12) Initializing personal best positions & fitnesses
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitnesses = np.full(num_particles, np.inf)

        # (Line 13) Fitness buffer for current positions
        self.fitnesses = np.full(num_particles, np.inf)

        # (Lines 14–15) Creating informant network for each particle
        self.informants = self._create_informant_network()

        # (Line 16) Initializing global best
        self.global_best_position = None
        self.global_best_fitness = np.inf

        # (Line 17) Fitness history for tracking
        self.fitness_history = []
    
    def _create_informant_network(self):
        network = []
        for i in range(self.num_particles):
            others = list(range(self.num_particles))
            selected = np.random.choice(
                others, self.num_informants, replace=False
            ).tolist()
            if i not in selected:
                selected.append(i)
            network.append(selected)
        return network

    def _informant_best(self, particle_index):
        best_fit = np.inf
        best_pos = None
        for idx in self.informants[particle_index]:
            if self.personal_best_fitnesses[idx] < best_fit:
                best_fit = self.personal_best_fitnesses[idx]
                best_pos = self.personal_best_positions[idx]
        return best_pos.copy()

    def _apply_bounds(self, x):
        # (Line 27) Ensuring positions remain within bounds
        return np.clip(x, self.bounds[0], self.bounds[1])

    def optimize(self, fitness_function, verbose=True):

        for iteration in range(self.max_iterations):

          
            # (Lines 12–15) Evaluating all particles & updating personal/global bests 
            for i in range(self.num_particles):
                f = fitness_function(self.positions[i])
                self.fitnesses[i] = f

                # Personal best
                if f < self.personal_best_fitnesses[i]:
                    self.personal_best_fitnesses[i] = f
                    self.personal_best_positions[i] = self.positions[i].copy()

                # Global best
                if f < self.global_best_fitness:
                    self.global_best_fitness = f
                    self.global_best_position = self.positions[i].copy()

          
            # (Lines 16–24) Determining new velocities
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

            # (Lines 25–26) Moving particles using step size e
            self.positions += self.e * self.velocities
            self.positions = self._apply_bounds(self.positions)

            # (Line 27) Recording global best fitness for plotting
            self.fitness_history.append(self.global_best_fitness)

            if verbose:
                avg_fit = np.mean(self.fitnesses)
                print(f"Iter {iteration:3d}: Best = {self.global_best_fitness:.6f}, Avg = {avg_fit:.6f}")

        # (Line 28) Returning best solution found
        return self.global_best_position, self.global_best_fitness

# TEST FUNCTIONS
def sphere_function(x):
    return np.sum(x ** 2)

# TEST RUN
if __name__ == "__main__":
    pso = PSO(num_particles=20,
              num_dimensions=5,
              num_informants=3,
              max_iterations=50,
              bounds=(-5.0, 5.0),
              alpha=0.7,
              beta=1.5,
              gamma=1.5,
              delta=0.5,
              step_size=0.5)

    best_pos, best_fit = pso.optimize(sphere_function, verbose=True)
    print(f"\nBest position found: {best_pos}")
    print(f"Best fitness: {best_fit:.6f}")
