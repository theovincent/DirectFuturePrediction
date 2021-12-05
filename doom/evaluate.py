import sys


def evaluate_cli(argvs=sys.argv[1:]):
    import argparse

    parser = argparse.ArgumentParser("Pipeline to evaluate a model to play doom")
    parser.add_argument(
        "-t",
        "--test",
        default=False,
        action="store_true",
        help="if given, path_data will be modified with the correct path to the data in my google drive, otherwise nothing happens, (default: False)",
    )
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     type=str,
    #     required=True,
    #     metavar="M",
    #     help="the model name (required)",
    #     choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "efficientnet"],
    # )
    # parser.add_argument(
    #     "-psw",
    #     "--path_starting_weights",
    #     type=str,
    #     default="ImageNet",
    #     metavar="PSW",
    #     help="the path to the starting weights, if None, takes random weights, 'output' will be added to the front (default: ImageNet)",
    # )
    # parser.add_argument(
    #     "-pd",
    #     "--path_data",
    #     type=str,
    #     required=True,
    #     metavar="PD",
    #     help="the path that leads to the data, 'bird_dataset' will be added to the front (required)",
    # )
    # parser.add_argument(
    #     "-nc",
    #     "--number_classes",
    #     type=int,
    #     default=20,
    #     metavar="NC",
    #     help="the number of classes to classify (default: 20)",
    # )
    # parser.add_argument(
    #     "-4D",
    #     "--classifier_4D",
    #     default=False,
    #     action="store_true",
    #     help="if given, a segmentation map will be added to the input, (default: False)",
    # )
    # parser.add_argument(
    #     "-fe",
    #     "--feature_extraction",
    #     default=False,
    #     action="store_true",
    #     help="if given, feature extraction will be performed, otherwise full training will be done, (default: False)",
    # )
    # parser.add_argument(
    #     "-bs", "--batch_size", type=int, default=64, metavar="BS", help="input batch size for training (default: 64)"
    # )
    # parser.add_argument(
    #     "-ne", "--n_epochs", type=int, default=1, metavar="NE", help="number of epochs to train (default: 10)"
    # )
    # parser.add_argument(
    #     "-lr",
    #     "--learning_rate",
    #     type=float,
    #     default=0.0005,
    #     metavar="LR",
    #     help="first learning rate before decreasing (default: 0.0005)",
    # )
    # parser.add_argument("-s", "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    # parser.add_argument(
    #     "-po",
    #     "--path_output",
    #     type=str,
    #     required=True,
    #     metavar="PO",
    #     help="folder where experiment outputs are located, 'output' will be added to the front (required)",
    # )
    args = parser.parse_args(argvs)
    print(args)


from __future__ import print_function
import sys

sys.path = ["../.."] + sys.path
from DFP.multi_experiment import MultiExperiment
import numpy as np
import time


def main(main_args):  ## main_args = "show" or "train"

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
    simulator_args = {}
    simulator_args["config"] = "../../maps/D1_basic.cfg"
    simulator_args["resolution"] = (84, 84)
    simulator_args["frame_skip"] = 4
    simulator_args["color_mode"] = "GRAY"
    simulator_args["maps"] = ["MAP01"]
    simulator_args["switch_maps"] = False
    # train
    simulator_args["num_simulators"] = 8

    ## Experience
    # Train experience
    train_experience_args = {}
    train_experience_args["memory_capacity"] = 20000
    train_experience_args["history_length"] = 1
    train_experience_args["history_step"] = 1
    train_experience_args["action_format"] = "enumerate"
    train_experience_args["shared"] = False

    # Test prediction experience
    test_prediction_experience_args = train_experience_args.copy()
    test_prediction_experience_args["memory_capacity"] = 1

    # Test policy experience
    test_policy_experience_args = train_experience_args.copy()
    test_policy_experience_args["memory_capacity"] = 55000

    ## Agent
    agent_args = {}

    # agent type
    agent_args["agent_type"] = "advantage"

    # preprocessing
    agent_args["preprocess_input_images"] = lambda x: x / 255.0 - 0.5
    agent_args["preprocess_input_measurements"] = lambda x: x / 100.0 - 0.5
    targ_scale_coeffs = np.expand_dims(
        (np.expand_dims(np.array([30.0]), 1) * np.ones((1, len(target_maker_args["future_steps"])))).flatten(), 0
    )
    agent_args["preprocess_input_targets"] = lambda x: x / targ_scale_coeffs
    agent_args["postprocess_predictions"] = lambda x: x * targ_scale_coeffs

    # agent properties
    agent_args["objective_coeffs_temporal"] = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0]
    agent_args["objective_coeffs_meas"] = [1.0]
    agent_args["random_exploration_schedule"] = lambda step: (0.02 + 145000.0 / (float(step) + 150000.0))
    agent_args["new_memories_per_batch"] = 8

    # net parameters
    agent_args["conv_params"] = np.array(
        [(32, 8, 4), (64, 4, 2), (64, 3, 1)], dtype=[("out_channels", int), ("kernel", int), ("stride", int)]
    )
    agent_args["fc_img_params"] = np.array([(512,)], dtype=[("out_dims", int)])
    agent_args["fc_meas_params"] = np.array([(128,), (128,), (128,)], dtype=[("out_dims", int)])
    agent_args["fc_joint_params"] = np.array(
        [(512,), (-1,)], dtype=[("out_dims", int)]
    )  # we put -1 here because it will be automatically replaced when creating the net
    agent_args["weight_decay"] = 0.00000

    # optimization parameters
    agent_args["batch_size"] = 64
    agent_args["init_learning_rate"] = 0.0001
    agent_args["lr_step_size"] = 250000
    agent_args["lr_decay_factor"] = 0.3
    agent_args["adam_beta1"] = 0.95
    agent_args["adam_epsilon"] = 1e-4
    agent_args["optimizer"] = "Adam"
    agent_args["reset_iter_count"] = False

    # directories
    agent_args["checkpoint_dir"] = "checkpoints"
    agent_args["log_dir"] = "logs"
    agent_args["init_model"] = ""
    agent_args["model_name"] = "predictor.model"
    agent_args["model_dir"] = time.strftime("%Y_%m_%d_%H_%M_%S")

    # logging and testing
    agent_args["print_err_every"] = 50
    agent_args["detailed_summary_every"] = 1000
    agent_args["test_pred_every"] = 0
    agent_args["test_policy_every"] = 7812
    agent_args["num_batches_per_pred_test"] = 0
    agent_args["num_steps_per_policy_test"] = (
        test_policy_experience_args["memory_capacity"] / simulator_args["num_simulators"]
    )
    agent_args["checkpoint_every"] = 10000
    agent_args["save_param_histograms_every"] = 5000
    agent_args["test_policy_in_the_beginning"] = True

    # experiment arguments
    experiment_args = {}
    experiment_args["num_train_iterations"] = 820000
    experiment_args["test_objective_coeffs_temporal"] = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    experiment_args["test_objective_coeffs_meas"] = np.array([1.0])
    experiment_args["test_random_prob"] = 0.0
    experiment_args["test_checkpoint"] = "checkpoints/2017_04_09_09_07_45"
    experiment_args["test_policy_num_steps"] = 2000
    experiment_args["show_predictions"] = False
    experiment_args["multiplayer"] = False

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


from __future__ import print_function
import numpy as np
from .future_target_maker import FutureTargetMaker
from .multi_doom_simulator import MultiDoomSimulator
from .multi_experience_memory import MultiExperienceMemory
from .future_predictor_agent_basic import FuturePredictorAgentBasic
from .future_predictor_agent_advantage import FuturePredictorAgentAdvantage
from .future_predictor_agent_advantage_nonorm import FuturePredictorAgentAdvantageNoNorm
from . import defaults
import tensorflow as tf
import scipy.misc
from . import util as my_util
import shutil

### Experiment with multi-head experience


class MultiExperiment:
    def __init__(
        self,
        target_maker_args={},
        simulator_args={},
        train_experience_args={},
        test_policy_experience_args={},
        agent_args={},
        experiment_args={},
    ):

        # set default values
        target_maker_args = my_util.merge_two_dicts(defaults.target_maker_args, target_maker_args)
        if isinstance(simulator_args, dict):
            simulator_args = my_util.merge_two_dicts(defaults.simulator_args, simulator_args)
        else:
            for n in range(len(simulator_args)):
                simulator_args[n] = my_util.merge_two_dicts(defaults.simulator_args, simulator_args[n])
        train_experience_args = my_util.merge_two_dicts(defaults.train_experience_args, train_experience_args)
        test_policy_experience_args = my_util.merge_two_dicts(
            defaults.test_policy_experience_args, test_policy_experience_args
        )
        agent_args = my_util.merge_two_dicts(defaults.agent_args, agent_args)
        experiment_args = my_util.merge_two_dicts(defaults.experiment_args, experiment_args)

        if not (experiment_args["args_file"] is None):
            print(" ++ Reading arguments from ", experiment_args["args_file"])
            with open(experiment_args["args_file"], "r") as f:
                input_args = my_util.json_load_byteified(f)

            for arg_name, arg_val in input_args.items():
                print(arg_name, arg_val)
                for k, v in arg_val.items():
                    locals()[arg_name][k] = v

        self.target_maker = FutureTargetMaker(target_maker_args)
        self.results_file = experiment_args["results_file"]
        self.net_name = experiment_args["net_name"]
        self.num_predictions_to_show = experiment_args["num_predictions_to_show"]
        agent_args["target_dim"] = self.target_maker.target_dim
        agent_args["target_names"] = self.target_maker.target_names

        if isinstance(simulator_args, list):
            # if we are given a bunch of different simulators
            self.multi_simulator = MultiDoomSimulator(simulator_args)
        else:
            # if we have to replicate a single simulator
            self.multi_simulator = MultiDoomSimulator([simulator_args] * simulator_args["num_simulators"])
        agent_args["discrete_controls"] = self.multi_simulator.discrete_controls
        agent_args["continuous_controls"] = self.multi_simulator.continuous_controls

        agent_args["objective_indices"], agent_args["objective_coeffs"] = my_util.make_objective_indices_and_coeffs(
            agent_args["objective_coeffs_temporal"], agent_args["objective_coeffs_meas"]
        )

        train_experience_args["obj_shape"] = (len(agent_args["objective_coeffs"]),)
        test_policy_experience_args["obj_shape"] = (len(agent_args["objective_coeffs"]),)
        self.train_experience = MultiExperienceMemory(
            train_experience_args, multi_simulator=self.multi_simulator, target_maker=self.target_maker
        )
        agent_args["state_imgs_shape"] = self.train_experience.state_imgs_shape
        agent_args["obj_shape"] = (len(agent_args["objective_coeffs"]),)
        agent_args["num_simulators"] = self.multi_simulator.num_simulators

        if "meas_for_net" in experiment_args:
            agent_args["meas_for_net"] = []
            for ns in range(self.train_experience.history_length):
                agent_args["meas_for_net"] += [
                    i + self.multi_simulator.num_meas * ns for i in experiment_args["meas_for_net"]
                ]  # we want these measurements from all timesteps
            agent_args["meas_for_net"] = np.array(agent_args["meas_for_net"])
        else:
            agent_args["meas_for_net"] = np.arange(self.train_experience.state_meas_shape[0])
        if len(experiment_args["meas_for_manual"]) > 0:
            agent_args["meas_for_manual"] = np.array(
                [
                    i + self.multi_simulator.num_meas * (self.train_experience.history_length - 1)
                    for i in experiment_args["meas_for_manual"]
                ]
            )  # current timestep is the last in the stack
        else:
            agent_args["meas_for_manual"] = []
        agent_args["state_meas_shape"] = [len(agent_args["meas_for_net"])]
        self.agent_type = agent_args["agent_type"]

        if agent_args["random_objective_coeffs"]:
            assert "fc_obj_params" in agent_args

        self.test_policy_experience = MultiExperienceMemory(
            test_policy_experience_args, multi_simulator=self.multi_simulator, target_maker=self.target_maker
        )
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)  # avoid using all gpu memory
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        if self.agent_type == "basic":
            self.ag = FuturePredictorAgentBasic(self.sess, agent_args)
        elif self.agent_type == "advantage":
            self.ag = FuturePredictorAgentAdvantage(
                self.sess, agent_args
            )  # inital design: concat meas and img, then 2 branches for adv and val
        elif self.agent_type == "advantage_nonorm":
            self.ag = FuturePredictorAgentAdvantageNoNorm(
                self.sess, agent_args
            )  # no normalizatio in the advantage stream
        else:
            raise Exception("Unknown agent type", self.agent_type)

        self.num_train_iterations = experiment_args["num_train_iterations"]
        _, self.test_objective_coeffs = my_util.make_objective_indices_and_coeffs(
            experiment_args["test_objective_coeffs_temporal"], experiment_args["test_objective_coeffs_meas"]
        )
        self.test_random_prob = experiment_args["test_random_prob"]
        self.test_checkpoint = experiment_args["test_checkpoint"]
        self.test_policy_num_steps = experiment_args["test_policy_num_steps"]

    def run(self, mode):
        shutil.copy("run_exp.py", "run_exp.py." + mode)
        if mode == "show":
            if not self.ag.load(self.test_checkpoint):
                print("Could not load the checkpoint ", self.test_checkpoint)
                return
            self.train_experience.head_offset = self.test_policy_num_steps + 1
            self.train_experience.log_prefix = "logs/log_test"
            self.ag.test_policy(
                self.multi_simulator,
                self.train_experience,
                self.test_objective_coeffs,
                self.test_policy_num_steps,
                random_prob=self.test_random_prob,
                write_summary=False,
                write_predictions=True,
            )
            self.train_experience.show(
                start_index=0,
                end_index=self.test_policy_num_steps * self.multi_simulator.num_simulators,
                display=True,
                write_imgs=False,
                preprocess_targets=self.ag.preprocess_input_targets,
                show_predictions=self.num_predictions_to_show,
                net_discrete_actions=self.ag.net_discrete_actions,
            )
        elif mode == "train":
            self.test_policy_experience.log_prefix = "logs/log"
            self.ag.train(
                self.multi_simulator,
                self.train_experience,
                self.num_train_iterations,
                test_policy_experience=self.test_policy_experience,
            )
        else:
            print("Unknown mode", mode)
