from solver import SparseSignal, DFTSolver, Verifier
import matplotlib.pyplot as plt
import numpy as np


class NoiseExperiment:
    """
    Runs an experiment to test error on nearly-sparse signals.
    Adds Gaussian noise to s-sparse signals and measures how recovery error grows as noise level increases.

    Attributes:
        n (int): Length of the signal.
        s (int): Sparsity level.
        min_noise (float): Minimum noise level to test.
        max_noise (float): Maximum noise level to test.
        noise_interval_count (int): Number of noise levels to test.
        runs_per_noise_level (int): Number of random signals per noise level.
    """
    def __init__(self, n, s, min_noise, max_noise, noise_interval_count, runs_per_noise_level):
        self.n = n
        self.s = s
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.noise_interval_count = noise_interval_count
        self.runs_per_noise_level = runs_per_noise_level
    
    def run(self):
        """
        This method runs the experiment with the given class configurations and it plots noise level vs average error.
        """
        noise_levels = []
        errors = []

        noise_levels = np.logspace(np.log10(self.min_noise), np.log10(self.max_noise), self.noise_interval_count)

        for level in noise_levels:
            curr_noise_errors = np.zeros(self.runs_per_noise_level)
            for i in range(len(curr_noise_errors)):
                signal = SparseSignal(self.n, self.s)
                signal.generate_noisy(noise_level=level)
                solver = DFTSolver(self.n, self.s, signal.dft_coefficients)
                solver.solve()
                verifier = Verifier(signal.x, solver.x_recovered)
                curr_noise_errors[i] = verifier.error()
            avg_error = np.average(curr_noise_errors)
            errors.append(avg_error)

        plt.plot(noise_levels, errors)
        plt.xlabel("Noise Level")
        plt.ylabel("Average Error (||x_original - x_new|| / ||x_original||)")
        plt.xscale("log")
        plt.title("Noise Level vs Recovery Error")
        plt.show()
    
    def plot_histogram(self, noise_level, trials=1000, ):
        errors = []
        for i in range(trials):
            signal = SparseSignal(self.n, self.s)
            signal.generate_noisy(noise_level=noise_level)
            solver = DFTSolver(self.n, self.s, signal.dft_coefficients)
            solver.solve()
            verifier = Verifier(signal.x, solver.x_recovered)
            errors.append(verifier.error())
        plt.hist(errors, bins=50)
        plt.xlabel("Average Error (||x_original - x_new|| / ||x_original||)")
        plt.xlim(0, np.percentile(errors, 99.9))
        plt.ylabel("Frequency")
        plt.show()


def main():
    """
    Calls the experiment to be ran.
    """

    # The following parameters can be altered.
    n = 10
    s = 2 # Make sure n > 2s
    min_noise = 1e-10
    max_noise = 1e-2
    noise_interval_count = 100
    runs_per_noise_level = 5

    experiment = NoiseExperiment(n,s,min_noise,max_noise,noise_interval_count,runs_per_noise_level)
    experiment.run()

    # Seperate experiment at one set noise_level
    experiment.plot_histogram(trials=10000, noise_level=1e-1)

if __name__ == "__main__":
    main()