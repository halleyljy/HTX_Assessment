ED Simulation
=============

1. Create a Python venv and install requirements:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run the simulation:
   python ED_Simulation.py

   This creates `output/ed_summary.csv`, `avg_wait.png`, `avg_los.png`,
   and `ED_sim_presentation.pptx`.

3. To experiment:
   - Edit parameters at top of ED_Simulation.py (arrival_rate_per_hour, staff counts, beds) or modify sim_settings.json if user is unfamiliar with Python
   - Re-run and compare CSVs
   - Use separate scripts to sweep parameters and collect results
