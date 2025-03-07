from .simulator.simulator_mcmd_exp import MCMDSimulator
from MCMD_Sim.utils import generate_init_conditions_closed_loop_inference_2choice
import numpy as np

if __name__ == '__main__':
    init_cond = generate_init_conditions_closed_loop_inference_2choice(
        ['red ball', 'blue ball'],
        1,
        ['results/test_mcmd/']
    )
    print(init_cond, 'init_cond')

    sim = MCMDSimulator('results/test_mcmd/', init_cond, 3)
    sim.precompute_trajectory()
    sim.run_simulation_to_completion()

    sim.export_plots()



    # sim.setup_simulation()

    # vel_cmd = np.array([0,0,0,0])
    # sim.vel_cmd_world = vel_cmd
    # updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
    # updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
    # print(updated_position, 'updated_pos')

    # init_forward = updated_position[0]
    # # controller = WaypointController(init_cond, Kp=0.15, threshold=0.03)
    # controller = AdaptivePIDController(init_cond)

    # unnormalized_vel_cmds = []
    # vel_cmd = np.array([1.0,1.0,0,0])
    # for step in range(5):
    #     # (b) Get the next velocity command from the controller
    #     # vel_cmd = controller.step(updated_position)

    #     # If done is set, the controller will give [0, 0, 0, 0].
    #     # You can also check controller.done explicitly:
    #     if controller.done:
    #         print(f"Controller finished path at step {step}")
    #         break

    #     # (c) Pass velocity command to the simulator
    #     updated_state, _, sim_finished = sim.dynamic_step_simulation(vel_cmd)
    #     if sim_finished:
    #         print(f"Simulator ended at step {step}")
    #         break

    #     # (e) Get new drone state in [x, y, z, yaw] form
    #     old_position = updated_position.copy()
    #     updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
    #     # controller.update_bayes(old_position, updated_position)

    #     # Debug prints
    #     print(f"Step {step}, Cmd: {vel_cmd}, Pos: {updated_position}")
    #     # updated_position -= [init_forward, 0, 0.6, sim.theta_environment]
    #     # print(updated_position, 'after subtraction')

    #     # (f) Optionally store velocity commands
    #     unnormalized_vel_cmds.append(vel_cmd)
        
    # np.savetxt("results/test/vel_cmds_unnorm.csv", np.array(unnormalized_vel_cmds), delimiter=",")



