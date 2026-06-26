from solver import SparseSignal, DFTSolver, Verifier
import numpy as np

class SparsityMismatchExperiment:
    """
    Runs an experiment where an s-sparse signal is recovered but the signal actually has less than s non-zero elements. 
    The same is done with a truly s-sparse signal. It is repeated multiple times and errors are averaged.
    """
    def __init__(self, n, s, true_s, trials):
        self.n = n
        self.s = s
        self.true_s = true_s
        self.trials = trials

    def run(self):
        errors = []
        under_sparse_errors = []

        for i in range(self.trials):
            signal = SparseSignal(self.n, self.s)
            signal.generate()
            solver = DFTSolver(self.n, self.s, signal.dft_coefficients)
            solver.solve()

            verifier = Verifier(signal.x, solver.x_recovered)
            errors.append(verifier.error())

            under_sparse_signal = SparseSignal(self.n, self.true_s)
            under_sparse_signal.generate()
            under_sparse_signal.compute_dft(self.s)

            under_sparse_solver = DFTSolver(self.n, self.s, under_sparse_signal.dft_coefficients)
            under_sparse_solver.solve()

            under_sparse_verifier = Verifier(under_sparse_signal.x, under_sparse_solver.x_recovered)
            under_sparse_errors.append(under_sparse_verifier.error())

        avg_error = np.average(errors)
        under_sparse_avg_error = np.average(under_sparse_errors)
        
        print(f"\nError (true {self.s}-sparse): {avg_error}\n")
        print(f"Error (actually {self.true_s}-sparse): {under_sparse_avg_error}\n")


def main():
    # The following parameters can be edited:
    n = 20
    s = 5
    true_s = 5
    trials = 100

    experiment = SparsityMismatchExperiment(
        n=n,
        s=s,
        true_s=true_s,
        trials=trials
    )

    experiment.run()


if __name__ == "__main__":
    main()