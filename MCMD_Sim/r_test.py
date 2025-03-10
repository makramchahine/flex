from simulator import MCMDSimulator
from simulator.utils import generate_closed_loop_1drone_2ball_env_init
import numpy as np

if __name__ == '__main__':
    init_cond = generate_closed_loop_1drone_2ball_env_init(command='to')
    print(init_cond, 'init_cond')

    sim = MCMDSimulator(init_cond, 3)
    sim.run_simulation_to_completion()
