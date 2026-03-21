import lzma 
import pickle
import time
from pathlib import Path
import numpy as np
import csv
import concurrent.futures
from tqdm import tqdm
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig

proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig()
traffic_agents_policy = LogReplayTrafficAgents(simulator.proposal_sampling)

def convert_one(metric_cache_path: str):
    with lzma.open(metric_cache_path, "rb") as f:
        metric_cache: MetricCache = pickle.load(f)
    new_metric_path = Path(metric_cache_path.replace('train_metric_cache_v2', 'train_metric_cache_v2_pruned'))
    new_metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_cache.file_path = new_metric_path
    
    # Simulate env and update observation
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory

    trajectory_states = [
        get_trajectory_as_array(pdm_trajectory, simulator.proposal_sampling, initial_ego_state.time_point),
    ]

    trajectory_states = np.stack(trajectory_states, axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)#32,41,11
    
    # infer traffic agents policy and update future observation
    # We all assume using log replay mode
    simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
        simulated_states[0], metric_cache
    )
    metric_cache.observation.update_detections_tracks(
        detection_tracks=simulated_agent_detections_tracks,
    )
    
    metric_cache.past_detections_tracks = None
    metric_cache.current_tracked_objects = None
    metric_cache.future_tracked_objects = None
    metric_cache.observation._detections_tracks = None

    metric_cache.dump()

if __name__ == "__main__":
    csv_path = '/scratch/linhan/train_metric_cache_v2/metadata/train_metric_cache_v2_metadata_node_0.csv'
    file_names: list[str] = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        file_names = [str(row['file_name']) for row in reader]

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for _ in tqdm(executor.map(convert_one, file_names, chunksize=16), total=len(file_names)):
            pass
    print(f"Completed pruning {len(file_names)} metric caches in {time.time() - start:.1f}s")
