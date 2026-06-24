import numpy as np

class SparseSignal:
    """
    Generates and stores an s-sparse signal its DFT coefficients. Has the option to generate a noisy signal with generate_noisy().
    """
    def __init__(self, n, s):
        self.n = n
        self.s = s
        self.x = None
        self.support = None
        self.dft_coefficients = None

    def generate(self):
        """
        Generate an s-sparse signal of length n
        """
        self.x = np.zeros(self.n, dtype=complex)

        non_sparse_idxs = np.random.choice(self.n, self.s, replace=False)
        self.support = non_sparse_idxs

        random_vals = np.random.randint(1, 10, size=self.s) + 1j * np.random.randint(1, 10, size=self.s)
        self.x[self.support] = random_vals

        self.compute_dft()
        

    def generate_noisy(self, noise_level=1e-6):
        """
        Generate an s-sparse signal of length n with small random noise added to every element
        """
        self.x = np.zeros(self.n, dtype=complex)

        non_sparse_idxs = np.random.choice(self.n, self.s, replace=False)
        self.support = non_sparse_idxs

        random_vals = np.random.randint(1, 10, size=self.s) + 1j * np.random.randint(1, 10, size=self.s)
        self.x[self.support] = random_vals

        # Adding random noise to every element
        self.x += np.random.normal(0, noise_level, self.n) + 1j * np.random.normal(0, noise_level, self.n)

        self.compute_dft()


    def compute_dft(self, s=None):
        """
        Computes the 2s DFT coefficients of the signal x
        """
        num_coeffs = s if s is not None else self.s
        self.dft_coefficients = np.zeros(2 * num_coeffs, dtype=complex)
        for j in range(0, 2 * num_coeffs):
            sum = 0
            for k in range(self.n):
                sum += self.x[k] * np.exp(-2j * np.pi * k * j / self.n)
            self.dft_coefficients[j] = sum

class DFTSolver:
    """
    Regenerates and stores a signal from inputted DFT coefficients.
    """
    def __init__(self, n, s, dft_coeffs):
        self.n = n
        self.s = s
        self.dft_coeffs = dft_coeffs
        self.M = None
        self.b = None
        self.q_hat = None
        self.q = None
        self.support = None
        self.x_recovered = None
    
    def form_system(self):
        """
        Defines and populates vector b and matrix M which contain inputted DFT coefficients.
        """
        self.M = np.zeros((self.s,self.s), dtype=complex) # Initializing a (s,s) matrix
        self.b = np.zeros(self.s, dtype=complex) # Initializing a (s,) vector such that b=Ax

        # Filling in M:
        for row in range(self.s):
            for col in range(self.s):
                self.M[row][col] = self.dft_coeffs[self.s - 1 + row - col]
        
        for i in range(self.s):
            self.b[i] = -self.dft_coeffs[self.s + i]

    def solve_system(self):
        """
        Solves the equation formed in the method form_system. The equation is as follows: M * (q_hat) = b where we solve for q_hat.
        
        Expected error: When running sparsity_mismatch_experiment.py, under-sparse signals can cause the np.linalg.solve() to fail due to no unique solution.
        """

        try:
            self.q_hat = np.linalg.solve(self.M, self.b)
        except np.linalg.LinAlgError as e:
            print("\nSolve failed:")
            print(e)
            print("\n--- Matrix Diagnostics ---")
            print("M shape:", self.M.shape)
            print("M rank :", np.linalg.matrix_rank(self.M))
            print("det(M) :", np.linalg.det(self.M))

    def reconstruct_q(self):
        """
        Reconstructs q based on the inverse DFT and evaluates for t=0,1,...,n-1
        """
        q_hat_full = np.zeros(self.n, dtype=complex)
        q_hat_full[0] = 1
        q_hat_full[1:self.s+1] = self.q_hat
        self.q = np.zeros(self.n, dtype=complex)

        for t in range(self.n):
            sum = 0
            for j in range(self.n):
                sum += q_hat_full[j] * np.exp(2j * np.pi * j * t / self.n)
            self.q[t] = sum / self.n

    def find_support(self):
        """
        Finds the support indices of x by first creating a vector of magitudes of q for t=0,1,...,n-1. Then, it chooses the s smallest magnitudes as the support of x.
        """
        self.support = []

        q_magnitudes = np.zeros(self.n)
        for t in range(self.n):
            q_magnitudes[t] = abs(self.q[t])
        self.support = np.argsort(q_magnitudes)[0:self.s]

    def recover_x(self):
        """
        Using the calculated support of x, it forms a s * s system of equations which is solved to recover x.
        """
        a = self.dft_coeffs[0:self.s]
        B = np.zeros((self.s, self.s), dtype=complex)

        for row in range(self.s):
            for col in range(self.s):
                B[row][col] = np.exp(-2j * np.pi * self.support[col] * row / self.n)
        c = np.linalg.solve(B, a)
        
        self.x_recovered = np.zeros(self.n, dtype=complex)

        for i in range(len(self.support)):
            self.x_recovered[self.support[i]] = c[i]

    def solve(self):
        """
        This is the main function in this class that runs everything in order from forming the initial system of equations to recovering the signal x.
        """
        self.form_system()
        self.solve_system()
        self.reconstruct_q()
        self.find_support()
        self.recover_x()

class Verifier:
    """
    A class used to verify that the recovered signal matches the original signal within a given tolerance. 
    """
    def __init__(self, original, recovered):
        self.original = original
        self.recovered = recovered

    def verify(self):
        """
        Checks that every element in the recovered signal is within a specific magnitude range from the original signal.
        """
        if np.allclose(self.original, self.recovered, atol=1e-6) == True:
            print("Pass: Near equality check")
        else:
            print("FAIL: Near equality check")

    def error(self):
        """
        Returns the L2 norm difference between the recovered and original signal.
        """
        #print(np.linalg.norm(self.recovered - self.original))
        return np.linalg.norm(self.recovered - self.original)

def main():
    """
    Calls the above classes to be ran for an s-sparse vector generated below.
    """
    n = 6
    s = 2

    signal = SparseSignal(n, s)
    signal.generate()

    solver = DFTSolver(n, s, signal.dft_coefficients)
    solver.solve()
    
    print(f"\nOriginal x: {np.round(signal.x, 6)}")
    print(f"Recovered x: {np.round(solver.x_recovered, 6)}")
    print(f"Support found: {solver.support}\n")


    verifier = Verifier(signal.x, solver.x_recovered)
    verifier.verify()
    print(f"Error: {verifier.error()}\n")

if __name__ == "__main__":
    main()

