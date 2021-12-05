from __future__ import print_function
import sys

sys.path = ["../.."] + sys.path
from DFP.multi_experiment import MultiExperiment
import numpy as np
import time


def main(main_args):
    ### Set all arguments
    ## Target maker
    target_maker_args = {
        "future_steps": [1, 2, 4, 8, 16, 32],
        "meas_to_predict": [0],
        "min_num_targs": 3,
        "rwrd_schedule_type": "exp",
        "gammas": [],
        "invalid_targets_replacement": "nan",
    }
    ## Simulator
    simulator_args = {
        "config": "../../maps/D1_basic.cfg",
        "resolution": (84, 84),
        "frame_skip": 4,
        "color_mode": "GRAY",
        "maps": ["MAP01"],
        "switch_maps": False,
        "num_simulators": 8,
        "game_args": "",
    }

    ## Experience
    # Train experience
    train_experience_args = {
        "memory_capacity": 20000,
        "history_length": 1,
        "history_step": 1,
        "action_format": "enumerate",
        "shared": False,
        "meas_statistics_gamma": 0.0,
        "num_prev_acts_to_return": 0,
    }
    # Test prediction experience
    test_prediction_experience_args = train_experience_args.copy()
    test_prediction_experience_args["memory_capacity"] = 1

    # Test policy experience
    test_policy_experience_args = train_experience_args.copy()
    test_policy_experience_args["memory_capacity"] = 55000

    ## Agent
    targ_scale_coeffs = np.expand_dims(
        (np.expand_dims(np.array([30.0]), 1) * np.ones((1, len(target_maker_args["future_steps"])))).flatten(), 0
    )
    agent_args = {
        "agent_type": "advantage",
        "preprocess_input_images": lambda x: x / 255.0 - 0.5,
        "preprocess_input_measurements": lambda x: x / 100.0 - 0.5,
        "preprocess_input_targets": lambda x: x / targ_scale_coeffs,
        "postprocess_predictions": lambda x: x * targ_scale_coeffs,
        "discrete_controls_manual": [],
        "opposite_button_pairs": [],
        "add_experiences_every": 1,
        "random_objective_coeffs": False,
        "objective_coeffs_distribution": "none",
        "objective_coeffs_temporal": [0.0, 0.0, 0.0, 0.5, 0.5, 1.0],
        "objective_coeffs_meas": [1.0],
        "random_exploration_schedule": lambda step: (0.02 + 145000.0 / (float(step) + 150000.0)),
        "new_memories_per_batch": 8,
        "conv_params": np.array(
            [(32, 8, 4), (64, 4, 2), (64, 3, 1)], dtype=[("out_channels", int), ("kernel", int), ("stride", int)]
        ),
        "fc_img_params": np.array([(512,)], dtype=[("out_dims", int)]),
        "fc_meas_params": np.array([(128,), (128,), (128,)], dtype=[("out_dims", int)]),
        "fc_joint_params": np.array(
            [(512,), (-1,)], dtype=[("out_dims", int)]
        ),  # we put -1 here because it will be automatically replaced when creating the net
        "fc_obj_params": None,
        "weight_decay": 0.00000,
        "batch_size": 64,
        "init_learning_rate": 0.0001,
        "lr_step_size": 250000,
        "lr_decay_factor": 0.3,
        "adam_beta1": 0.95,
        "adam_epsilon": 1e-4,
        "optimizer": "Adam",
        "reset_iter_count": False,
        "clip_gradient": 0,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "init_model": "",
        "model_name": "predictor.model",
        "model_dir": time.strftime("%Y_%m_%d_%H_%M_%S"),
        "print_err_every": 50,
        "detailed_summary_every": 1000,
        "test_pred_every": 0,
        "test_policy_every": 7812,
        "num_batches_per_pred_test": 0,
        "num_steps_per_policy_test": (
            test_policy_experience_args["memory_capacity"] / simulator_args["num_simulators"]
        ),
        "checkpoint_every": 10000,
        "save_param_histograms_every": 5000,
        "test_policy_in_the_beginning": True,
    }

    # experiment arguments
    experiment_args = {
        "num_train_iterations": 820000,
        "test_objective_coeffs_temporal": np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0]),
        "test_objective_coeffs_meas": np.array([1.0]),
        "test_random_prob": 0.0,
        "test_checkpoint": "checkpoints/2017_04_09_09_07_45",
        "test_init_policy_prob": 0.0,
        "test_policy_num_steps": 2000,
        "show_predictions": False,
        "multiplayer": False,
        "meas_for_manual": [],  # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7]
        "results_file": "results.txt",
        "net_name": "unknown_net",
        "num_predictions_to_show": 10,
        "args_file": None,
    }

    # Create and run the experiment

    experiment = MultiExperiment(
        target_maker_args=target_maker_args,
        simulator_args=simulator_args,
        train_experience_args=train_experience_args,
        test_policy_experience_args=test_policy_experience_args,
        agent_args=agent_args,
        experiment_args=experiment_args,
    )

    experiment.run(main_args[0])


if __name__ == "__main__":
    main(sys.argv[1:])
