"""
Simulation functions for ED model.
- run_one_replication(params, seed)
- run_scenario(params, scenario_name, outdir)

Outputs are saved into outdir:
- ed_summary.csv (one row per replication)
- wait_samples.csv (each patient wait times)
- los_samples.csv (each patient LOS)
- queue_samples.csv (time, bed_queue_len, main_doc_queue_len, main_nurse_queue_len, fast_queue_len, beds_in_use)
- sim_settings.json

"""
import os
import json
import random
import simpy
import numpy as np
import pandas as pd
from datetime import datetime

# ------------------ Default Distributions / Parameters ------------------
# You may override these via scenario.txt
DEFAULTS = {
    'scenario_name': 'smoke_test',
    'replications': 12,
    'sim_days': 14,
    'arrival_rate_per_hour': 12.0,
    'p_fast': 0.35,
    'p_need_lab_fast': 0.10,
    'p_need_lab_main': 0.30,
    'p_admit_main': 0.2,
    'fast_doctors': 1,
    'fast_nurses': 1,
    'main_doctors': 5,
    'main_nurses': 10,
    'ed_treatment_beds': 25,
    'seed_base': 1000,
    'queue_sample_interval_min': 30
}

SCENARIO_RUNS = {
    'baseline':{
        'arrival_rate_per_hour': 11.0,
        'fast_doctors': 1,
        'fast_nurses': 1,
        'main_doctors': 5,
        'main_nurses': 10,
        'ed_treatment_beds': 20,
    },
    'new_baseline':{
        'arrival_rate_per_hour': 12.0,
        'fast_doctors': 1,
        'fast_nurses': 1,
        'main_doctors': 5,
        'main_nurses': 10,
        'ed_treatment_beds': 20,
    },
    'add_fast_track':{
        'arrival_rate_per_hour': 12.0,
        'fast_doctors': 2,
        'fast_nurses': 3,
        'main_doctors': 5,
        'main_nurses': 10,
        'ed_treatment_beds': 20,
    },
    'add_main_ED':{
        'arrival_rate_per_hour': 12.0,
        'fast_doctors': 1,
        'fast_nurses': 1,
        'main_doctors': 6,
        'main_nurses': 12,
        'ed_treatment_beds': 20,
    },
    'add_ED_beds':{
        'arrival_rate_per_hour': 12.0,
        'fast_doctors': 1,
        'fast_nurses': 1,
        'main_doctors': 5,
        'main_nurses': 10,
        'ed_treatment_beds': 25,
    }
}

# ------------------ Helper function to generate delay times ------------------
def fast_service_time():
    """return value in minutes"""
    return max(3, random.gauss(10, 5))

def main_service_time():
    """return value in minutes"""
    return max(5, random.gauss(30, 10))

def fast_lab_time():
    """return value in minutes"""
    return max(2, random.expovariate(1 / 10.0))

def main_lab_time():
    """return value in minutes"""
    return max(2, random.expovariate(1 / 30.0))

def transfer_delay():
    """return value in minutes"""
    # mean about 8 hours
    return max(15, random.expovariate(1 / (8 * 60)))

# ------------------ Stats collector ------------------

class Collector:
    def __init__(self):
        self.waits = []          # per-patient wait times (start of service - arrival)
        self.los = []            # length of stay per patient (end of service - arrival)
        self.num_discharged = 0
        self.num_admitted = 0
        self.boarding_times = []
        self.queue_samples = []  # queue sameple: simulation_time_now, bed_queue_len, main_doc_queue_len, main_nurse_queue_len, fast_queue_len, beds_in_use

# ------------------ Patient process ------------------

def patient_process(env:simpy.Environment, params: dict, resources: dict, collector: Collector):
    t_arr:float = env.now
    p_fast:float = params['p_fast']
    p_need_lab_fast:float = params['p_need_lab_fast']
    p_need_lab_main:float = params['p_need_lab_main']
    p_admit:float = params['p_admit_main']

    # Decide route
    if random.random() < p_fast:
        # Fast Track: needs both doctor + nurse
        with resources['fast_doctor'].request() as rd, resources['fast_nurse'].request() as rn:
            yield rd & rn
            wait:float = env.now - t_arr
            collector.waits.append(wait)
            
            yield env.timeout(fast_service_time())
            
        # lab
        if random.random() < p_need_lab_fast:
            # lab test: needs nurse
            with resources['fast_nurse'].request() as rn:
                yield rn
                yield env.timeout(fast_lab_time())

            # review lab test result: needs doctor + nurse
            with resources['fast_doctor'].request() as rd, resources['fast_nurse'].request() as rn:
                yield rd & rn
                yield env.timeout(fast_service_time())
            
        collector.los.append(env.now - t_arr)
        collector.num_discharged += 1
        return
    
    # Main ED: needs both doctor + nurse
    with resources['main_doctor'].request() as rd, resources['main_nurse'].request() as rn:
        yield rd & rn
        wait:float = env.now - t_arr
        collector.waits.append(wait)
        
        yield env.timeout(main_service_time())
        
    if random.random() < p_need_lab_main:
        # lab test: needs nurse
        with resources['main_nurse'].request() as rn:
            yield rn
            yield env.timeout(main_lab_time())
        
        # review lab test result: needs doctor + nurse
        with resources['main_doctor'].request() as rd, resources['main_nurse'].request() as rn:
            yield rd & rn
            yield env.timeout(main_service_time())
    
    if not random.random() < p_admit:
        # discharged
        collector.num_discharged += 1
        collector.los.append(env.now - t_arr)
    else:
        #admitted
        with resources['beds'].request() as bedreq:
            yield bedreq  # wait until a bed is available
            collector.num_admitted += 1
            boarding_start = env.now
            
            # remain in bed until transfer
            yield env.timeout(transfer_delay())
            collector.boarding_times.append(env.now - boarding_start)
            collector.los.append(env.now - t_arr)
    
# ------------------ Arrival generator ------------------

def arrivals(env:simpy.Environment, params: dict, resources: dict, collector: Collector):
    lam:float = params['arrival_rate_per_hour'] / 60.0
    while True:
        ia = random.expovariate(lam)
        yield env.timeout(ia)
        env.process(patient_process(env, params, resources, collector))

# ------------------ Monitor ------------------

def monitor_queues(env:simpy.Environment, params: dict, resources: dict, collector: Collector):
    interval:int = params.get('queue_sample_interval_min', DEFAULTS['queue_sample_interval_min'])
    while True:
        beds_in_use = resources['beds'].count
        bed_queue = len(resources['beds'].queue)
        main_doc_queue = len(resources['main_doctor'].queue)
        main_nurse_queue = len(resources['main_nurse'].queue)
        fast_q = len(resources['fast_doctor'].queue) + len(resources['fast_nurse'].queue)
        collector.queue_samples.append({
                'time_min': env.now,
                'bed_queue_len': bed_queue,
                'main_doc_queue_len': main_doc_queue,
                'main_nurse_queue_len': main_nurse_queue,
                'fast_queue_len': fast_q,
                'beds_in_use': beds_in_use
            })
        yield env.timeout(interval)

# ------------------ Single replication runner ------------------

def run_one_replication(params: dict, seed: int) -> Collector:
    random.seed(seed)
    np.random.seed(seed)
    sim_minutes = int(params['sim_days'] * 24 * 60)

    env = simpy.Environment()
    collector = Collector()
    
    # resources based on parameters
    resources = {
        'fast_doctor': simpy.Resource(env, capacity = params['fast_doctors']),
        'fast_nurse': simpy.Resource(env, capacity = params['fast_nurses']),
        'main_doctor': simpy.Resource(env, capacity = params['main_doctors']),
        'main_nurse': simpy.Resource(env, capacity = params['main_nurses']),
        'beds': simpy.Resource(env, capacity = params['ed_treatment_beds'])
    }
    
    # processes
    env.process(arrivals(env, params, resources, collector))
    env.process(monitor_queues(env, params, resources, collector))
    env.run(until = sim_minutes)

    return collector

# ------------------ Scenario runner ------------------

def run_scenario(scenario: dict, outdir: str) -> None:
    """
    takes in scenario settings
    outputs simulation data to directory
    """
    os.makedirs(outdir, exist_ok = True)
    # merge defaults with scenario
    params:dict = DEFAULTS.copy()
    params.update(scenario)

    rep = int(params.get('replications', DEFAULTS['replications']))
    seed_base = int(params.get('seed_base', DEFAULTS['seed_base']))

    rows, wait_rows, los_rows, queue_rows = ([] for _ in range(4))
    for rep_count in range(rep):
        seed:int = seed_base + rep_count
        coll:Collector = run_one_replication(params, seed)
        
        # summarise data from one replication into a row
        simulation_result = {
            'replication': rep_count + 1,
            'avg_wait_min': float(np.mean(coll.waits)) if coll.waits else 0.0,
            'p90_wait_min': float(np.percentile(coll.waits, 90)) if coll.waits else 0.0,
            'avg_los_min': float(np.mean(coll.los)) if coll.los else 0.0,
            'p90_los_min': float(np.percentile(coll.los, 90)) if coll.los else 0.0,
            'num_discharged': int(coll.num_discharged),
            'num_admitted': int(coll.num_admitted),
            'avg_boarding_min': float(np.mean(coll.boarding_times)) if coll.boarding_times else 0.0
        }
        
        rows.append(simulation_result)
        
        for w in coll.waits:
            wait_rows.append({'replication': rep_count + 1, 'wait_min': float(w)})
        
        for l in coll.los:
            los_rows.append({'replication': rep_count + 1, 'los_min': float(l)})
        
        for q in coll.queue_samples:
            # each queue sample includes simulation_time_now, bed_queue_len, doc_queue_len, nurse_queue_len, fast_queue_len, beds_in_use
            queue_rows.append({'replication': rep_count + 1, **q})
    
    # write CSVs
    df_summary = pd.DataFrame(rows)
    df_waits = pd.DataFrame(wait_rows)
    df_los = pd.DataFrame(los_rows)
    df_queue = pd.DataFrame(queue_rows)

    df_summary.to_csv(os.path.join(outdir, 'ed_summary.csv'), index = False)
    df_waits.to_csv(os.path.join(outdir, 'wait_samples.csv'), index = False)
    df_los.to_csv(os.path.join(outdir, 'los_samples.csv'), index = False)
    df_queue.to_csv(os.path.join(outdir, 'queue_samples.csv'), index = False)

    # simulation name, parameter and timestamp for future reference
    sim_settings = {
        'scenario_name': params.get('scenario_name', DEFAULTS['scenario_name']),
        'generated_at': str(datetime.now()),
        'params': params
    }
    with open(os.path.join(outdir, 'sim_settings.json'), 'w') as f:
        json.dump(sim_settings, f, indent = 2)

    print(f"Wrote outputs to {outdir}")


# ------------------ Setting scenario if user input is off ------------------

def setting_scenario(scenario: dict) -> dict:
    if not isinstance(scenario, dict):
        return DEFAULTS

    if not isinstance(scenario.get('scenario_name', False), str):
        scenario['scenario_name'] = DEFAULTS['scenario_name']
    
    if not isinstance(scenario.get('replications', False), int):
        scenario['replications'] = DEFAULTS['replications']
    
    if not isinstance(scenario.get('sim_days', False), int):
        scenario['sim_days'] = DEFAULTS['sim_days']

    if not isinstance(scenario.get('arrival_rate_per_hour', False), float):
        scenario['arrival_rate_per_hour'] = DEFAULTS['arrival_rate_per_hour']
    
    if not isinstance(scenario.get('p_fast', False), float):
        scenario['p_fast'] = DEFAULTS['p_fast']
    
    if not isinstance(scenario.get('p_need_lab', False), float):
        scenario['p_need_lab'] = DEFAULTS['p_need_lab']

    if not isinstance(scenario.get('p_admit_main', False), float):
        scenario['p_admit_main'] = DEFAULTS['p_admit_main']

    if not isinstance(scenario.get('fast_doctors', False), int):
        scenario['fast_doctors'] = DEFAULTS['fast_doctors']

    if not isinstance(scenario.get('fast_nurses', False), int):
        scenario['fast_nurses'] = DEFAULTS['fast_nurses']
    
    if not isinstance(scenario.get('main_doctors', False), int):
        scenario['main_doctors'] = DEFAULTS['main_doctors']

    if not isinstance(scenario.get('main_nurses', False), int):
        scenario['main_nurses'] = DEFAULTS['main_nurses']
    
    if not isinstance(scenario.get('ed_treatment_beds', False), int):
        scenario['ed_treatment_beds'] = DEFAULTS['ed_treatment_beds']
    
    if not isinstance(scenario.get('lab_servers', False), int):
        scenario['lab_servers'] = DEFAULTS['lab_servers']
    
    if not isinstance(scenario.get('seed_base', False), int):
        scenario['seed_base'] = DEFAULTS['seed_base']

    if not isinstance(scenario.get('queue_sample_interval_min', False), int):
        scenario['queue_sample_interval_min'] = DEFAULTS['queue_sample_interval_min']

    return scenario

# ------------------ run simulation ------------------

if __name__ == '__main__':
    directory:str = 'output\\smoke_test' #output directory to create results of simulation
    if not os.path.isdir(directory):
        directory = os.getcwd()
    
    example:dict = DEFAULTS.copy()
    setting_file = directory + "\\settings.json"
    if os.path.exists(setting_file):
        with open(setting_file, "r") as f:
            example:dict = json.load(f)
    example = setting_scenario(example)

    run_scenario(example, directory + '\\' + example['scenario_name'])

    for scenario, param in enumerate(SCENARIO_RUNS):
        example['scenario_name'] = scenario
        for key, val in enumerate(param):
            example[key] = param[key]
        run_scenario(example, directory + '\\' + example['scenario_name'])