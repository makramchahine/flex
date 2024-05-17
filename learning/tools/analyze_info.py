import argparse
import pickle
import glob
import numpy as np
from enum import Enum


class Behavior(Enum):
    LANE_FOLLOWING = 0
    AVOIDANCE = 1
    RECOVERY = 2
    

class BehaviorCriteria:
    DISTANCE_THRESHOLD = 10
    # DISTANCE_THRESHOLD_LANE_FOLLOWING_AFTER_RECOVERY = 12 # must set to the same as distance threshold otherwise will cause unhandle case
    MAX_EPISODE_LEN = 200

    @classmethod
    def print(cls):
        print_str = f"=== {cls.__name__} ===\n"
        for attribute, value in cls.__dict__.items():
            if not attribute.startswith('__') and isinstance(value, (int, float, list, dict)):
                print_str += f"{attribute} = {value}\n"
        print(print_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-pattern", type=str, required=True)
    parser.add_argument("--use-hard-behavior", action="store_true", default=False)
    args = parser.parse_args()
    
    print("")
    BehaviorCriteria.print()

    for pkl_path in sorted(glob.glob(args.pkl_pattern)):
        print("")
        print(pkl_path)
        with open(pkl_path, "rb") as f:
            info = pickle.load(f)
            
        if False:
            sample_info = info[0][0]
            print("Info keys: ", list(sample_info[sample_info["ego_agent_id"]].keys()))
        
        behavior = get_behavior(info)
        data_ego = get_relevant_data(info, is_ego=True)
        
        fail_at = []
        fail_ratio = []
        for ep_i in range(len(behavior)):
            fail_behavior = "None"
            for step_i in range(len(behavior[ep_i])):
                behavior_step = behavior[ep_i][step_i]
                
                done_info_step = dict()
                for done_info_key in ["trace_done", "done", "out_of_lane", "exceed_max_rot", "crashed"]:
                    done_info_step[done_info_key] = data_ego[done_info_key][ep_i][step_i]
                trace_done = done_info_step.pop("trace_done") # exclude this since this is not actually a failure
                
                if np.any(list(done_info_step.values())):
                    fail_behavior = behavior_step
                    assert step_i == (len(behavior[ep_i]) - 1), "Failure doesn't end at the last step"
            
            fail_at.append(fail_behavior)   
            fail_ratio.append(1 - (step_i / BehaviorCriteria.MAX_EPISODE_LEN if not trace_done else 1.))
        fail_at = np.array(fail_at)
        fail_ratio = np.array(fail_ratio)
        
        for behavior in [Behavior.LANE_FOLLOWING, Behavior.AVOIDANCE, Behavior.RECOVERY]:
            if args.use_hard_behavior:
                fail_at_behavior = fail_at == behavior
            else:
                fail_at_behavior = (fail_at == behavior).astype(float) * fail_ratio
            print(f"Failure rate at {behavior.name}: {np.mean(fail_at_behavior)}")
        print("========================")


def get_behavior(info):
    out = []
    for info_ep in info:
        out.append([])
        behavior_prev = Behavior.LANE_FOLLOWING
        for info_step in info_ep:
            ego_agent_id = info_step["ego_agent_id"]
            info_step_ego = info_step[ego_agent_id]
            
            ado_agent_id = [k for k in info_step.keys() if k not in [info_step["ego_agent_id"], "ego_agent_id"]][0]
            info_step_ado = info_step[ado_agent_id]
            
            ego_is_front = info_step_ego["frame_number"] > info_step_ado["frame_number"]
            dist = np.linalg.norm(info_step_ado["ego_dynamics"][:2] - info_step_ego["ego_dynamics"][:2])
            is_close = dist <= BehaviorCriteria.DISTANCE_THRESHOLD
            # is_close_after_recovery = dist <= BehaviorCriteria.DISTANCE_THRESHOLD_LANE_FOLLOWING_AFTER_RECOVERY
            if ego_is_front and is_close and (behavior_prev in [Behavior.AVOIDANCE, Behavior.RECOVERY]):
                behavior = Behavior.RECOVERY
            elif is_close and (behavior_prev in [Behavior.LANE_FOLLOWING, Behavior.AVOIDANCE]):
                behavior = Behavior.AVOIDANCE
            elif (not ego_is_front) and (not is_close) and (behavior_prev in [Behavior.LANE_FOLLOWING]):
                behavior = Behavior.LANE_FOLLOWING # lane following before recovery
            elif ego_is_front and (not is_close) and (behavior_prev in [Behavior.RECOVERY, Behavior.LANE_FOLLOWING]):
                behavior = Behavior.LANE_FOLLOWING # lane following after recovery
            else:
                raise ValueError(f"Unhandled case: ego_is_front={ego_is_front}, is_close={is_close}, behavior_prev={behavior_prev}")
            out[-1].append(behavior)
            
            behavior_prev = behavior

    return out


def get_relevant_data(info, is_ego):
    relevant_keys = [
        "ego_dynamics", "human_dynamics", "relative_state",
        "trace_done", "done", "out_of_lane", "exceed_max_rot", "crashed",
    ]
    data = dict()
    for key in relevant_keys:
        data[key] = unwrap_info(info, key, is_ego=is_ego)
    
    return data


def unwrap_info(info, key, is_ego, to_numpy=False):
    out = []
    for info_ep in info:
        out.append([])
        for info_step in info_ep:
            if is_ego:
                agent_id = info_step["ego_agent_id"]
            else:
                agent_id = [k for k in info_step.keys() if k not in [info_step["ego_agent_id"], "ego_agent_id"]][0]
            info_step_agent = info_step[agent_id]
            out[-1].append(info_step_agent[key])
    if to_numpy:
        out = np.array(out)
    
    return out


if __name__ == "__main__":
    main()
