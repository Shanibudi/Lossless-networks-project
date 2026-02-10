# PFC Cyclic Buffer Dependency Simulator

This repository contains a Python-based simulator that demonstrates how **Priority Flow Control (PFC)** can lead to **cyclic buffer dependency (CBD)** behavior in lossless networks, including buffer buildup, link pauses, and deadlock.

The simulator models a simplified leaf-spine topology where each switch has a single shared buffer, and packets are injected step-by-step according to predefined traffic flows. PFC is triggered whenever a downstream buffer crosses a threshold, potentially creating a circular wait condition.

The project includes two scenarios:

**Scenario 1:** Cyclic dependency without deadlock  
**Scenario 2:** Same dependency with an extra flow causing deadlock  


## Dependencies

Install the required packages using:

```bash
pip install numpy matplotlib networkx
```

## How to run the simulation script

You can run:

```bash
python pfc_cyclic_dependency_sim.py 
```

Optional flags:

- `--scenario {1,2,both}` - select which scenario to run (default: both)
- `--steps N`  (default: 30) — number of simulation steps.
- `--out_dir PATH`  (default: `.../fattree_lossless_networks/plots`)
- `--reports_dir PATH` (default: `.../fattree_lossless_networks/reports`)


## Output 
Running the script generates:
	•	`/plots` directory containing topology and simulation plots for each scenario
	•	`/reports` directory containing text summaries of each scenario execution
