from solver import SparseSignal, DFTSolver, Verifier
import numpy as np
import matplotlib.pyplot as plt

class ExtraMeasurementsExperiment:
    """
    Tests whether using more than 2s DFT measurements improves recovery of noisy s-sparse signals.
    For each value of r from 0 to r_max, generates noisy signals and recovers them using 2*(s+r) 
    measurements instead of the minimum 2s. Plots average recovery error vs r.
    """
    def __init__(self, n, s, trials, r_max):
        self.n = n
        self.s = s
        self.trials = trials
        self.r_max = r_max

    def run(self):
        errors = []
        r_values = []
        for r in range(self.r_max):
            r_values.append(r)
            signal_errors = []
            for i in range(self.trials):
                signal = SparseSignal(self.n, self.s)
                signal.generate_noisy(1e-4)
                signal.compute_dft(self.s + r)
                
                solver = DFTSolver(self.n, self.s + r, signal.dft_coefficients, true_s=self.s)
                solver.solve()

                verifier = Verifier(signal.x, solver.x_recovered)
                signal_errors.append(verifier.error())
            errors.append(np.average(signal_errors))
        
        plt.plot(r_values, errors, 'o')
        plt.xlabel("R values")
        plt.ylabel("Average Error")
        plt.title("R value vs Error")
        plt.show()


def main():
    # Feel free to adjust the following parameters:
    n = 60
    s = 3
    trials = 2000
    r_max = 10

    experiment = ExtraMeasurementsExperiment(n, s, trials, r_max)
    experiment.run()

if __name__ == "__main__":
    main()