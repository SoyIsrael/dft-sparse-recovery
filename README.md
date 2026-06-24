## How to Start it

1. Clone the repo and navigate to the project folder:
```bash
git clone https://github.com/SoyIsrael/dft-sparse-recovery.git
cd dft-sparse-recovery
```

2. Set up the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib # Installing necessary packages
```

3. Run the solver:
```bash
python3 solver.py
```

4. Run the noise experiment:
```bash
python3 noise_experiment.py
```

5. Run the sparsity mismatch experiment:
```bash
python3 sparsity_mismatch_experiment.py
```