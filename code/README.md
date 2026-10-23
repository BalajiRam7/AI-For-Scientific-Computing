# Homework 1

The PDF handout is at `handout/hw1_student.pdf` and
the LaTeX template is at `handout/hw1_student.tex`.

This package supplies starter code for the coding
section (Questions 11–13). Run the commands below
from the package directory. A GPU is not required.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To match the tested versions (Python 3.12) exactly,
use `requirements-tested.txt` instead of `requirements.txt`.

## Complete the five TODOs

| Question | File                   | Functions to complete                                    |
| -------- | ---------------------- | -------------------------------------------------------- |
| 12       | `code/starter_pinn.py` | `pde_residual`, `pinn_loss`, `sample_points`             |
| 13       | `code/starter_fno.py`  | `SpectralConv1d.forward`, `FNO1d.forward` |

The PINN starter has three TODOs and the FNO starter has two.
The FNO training function `train_epoch` is supplied. Neural training commands
raise `NotImplementedError` when they reach an unfinished TODO.

After completing the TODOs, run the small correctness checks with
`python -m unittest discover -s tests -p test_student.py -v`. They check shapes,
sampling domains, and loss terms against the handout's specification, not your
numerical results.

## Run the required experiments

Question 11: run the two finite-difference Euler solvers and reference checks.

```bash
python3 code/run_baselines.py --out results/classical
python code/plot_results.py --root results --section classical
```

Question 12: run three uniform-sampling seeds and one
importance-weighted focused-sampling intervention.

```bash
python code/starter_pinn.py --seed 0 --out results/pinn_seed0
python code/starter_pinn.py --seed 1 --out results/pinn_seed1
python code/starter_pinn.py --seed 2 --out results/pinn_seed2
python code/starter_pinn.py --seed 0 --sampling focused --out results/pinn_focused
python code/plot_results.py --root results --section pinn
```

Question 13: generate the training, validation, and test examples, then
train the FNO. Its driver selects the checkpoint by validation error,
evaluates resolution transfer, and runs the `sin(8*pi*x)` stress test.

```bash
python code/generate_data.py --out results/burgers_data.npz
python code/starter_fno.py --data results/burgers_data.npz --seed 0 --out results/fno
python code/plot_results.py --root results --section fno
```

After those runs, collect the tables, measure
the query costs, and plot the cost comparison:

```bash
python code/collect_results.py --root results
python code/plot_results.py --root results --section cost
```

## Find the outputs

- Each training directory contains `history.csv`, `model.pt` and `metrics.json`.
  - PINN directories also contain `prediction.npz`
  - The FNO directory contains `validation_prediction.npz`,
    `test_prediction_n{64,128,256,512,1024}.npz` and `stress_sine_k8.npz`.
- `results/figures/` contains the plots as PNG and PDF files.
- `results/classical/convergence.csv` contains the classical error table.
- `results/RESULTS.md` collects the requested numerical results.
- `results/cost_calibration.json` records the accuracy comparisons and timings.

Include the requested tables, figures, and interpretations
with your answers to the corresponding questions.
