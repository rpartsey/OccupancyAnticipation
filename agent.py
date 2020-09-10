import argparse
import random
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

import habitat
from occant_baselines.config.default import get_config
from occant_baselines.rl.ans import ActiveNeuralSLAMNavigator
from occant_baselines.models.occant import OccupancyAnticipator
from occant_baselines.rl.policy_utils import OccupancyAnticipationWrapper
from occant_baselines.models.mapnet import DepthProjectionNet
from habitat_baselines.common.utils import batch_obs
from occant_utils.common import add_pose
from einops import rearrange


class OccAntAgent(habitat.Agent):
    def __init__(self, config: habitat.Config):
        # Match configs for different parts of the model as well as the simulator
        self.config = config
        self._synchronize_configs(self.config)

        self._POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS  # check this line

        random.seed(config.PYT_RANDOM_SEED)
        np.random.seed(config.PYT_RANDOM_SEED)
        torch.manual_seed(config.PYT_RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))

        ckpt_dict = torch.load(config.EVAL_CKPT_PATH_DIR, map_location="cpu")

        if config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = config.clone()

        self.ans_cfg = config.RL.ANS
        occ_cfg = self.ans_cfg.ans_cfg.OCCUPANCY_ANTICIPATOR
        mapper_cfg = self.ans_cfg.ans_cfg.MAPPER

        # Create occupancy anticipation model
        occupancy_model = OccupancyAnticipationWrapper(
            OccupancyAnticipator(occ_cfg), mapper_cfg.map_size, (128, 128)
        )

        # Create ANS model
        self.ans_net = ActiveNeuralSLAMNavigator(self.ans_cfg, occupancy_model)
        self.mapper = self.ans_net.mapper
        self.local_actor_critic = self.ans_net.local_policy

        # Create depth projection model to estimate visible occupancy
        self.depth_projection_net = DepthProjectionNet(self.ans_cfg.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION)

        # Set to device
        self.mapper.to(self.device)
        self.local_actor_critic.to(self.device)
        self.depth_projection_net.to(self.device)

        # Convert the state_dict of mapper_agent to mapper
        mapper_dict = {
            k.replace("mapper.", ""): v
            for k, v in ckpt_dict["mapper_state_dict"].items()
        }
        # Converting the state_dict of local_agent to just the local_policy.
        local_dict = {
            k.replace("actor_critic.", ""): v
            for k, v in ckpt_dict["local_state_dict"].items()
        }
        self.mapper.load_state_dict(mapper_dict, strict=False)
        self.local_actor_critic.load_state_dict(local_dict)


    def _synchronize_configs(self, config):
        config.defrost()
        config.RL.ANS.PLANNER.nplanners = config.NUM_PROCESSES
        config.RL.ANS.MAPPER.thresh_explored = config.RL.ANS.thresh_explored
        config.RL.ANS.pyt_random_seed = config.PYT_RANDOM_SEED
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.pyt_random_seed = config.PYT_RANDOM_SEED
        # Compute the EGO_PROJECTION options based on the
        # depth sensor information and agent parameters.
        map_size = config.RL.ANS.MAPPER.map_size
        map_scale = config.RL.ANS.MAPPER.map_scale
        min_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        width = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        hfov_rad = np.radians(float(hfov))
        vfov_rad = 2 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
        vfov = np.degrees(vfov_rad).item()
        camera_height = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION[1]
        height_thresholds = [0.2, 1.5]
        # Set the EGO_PROJECTION options
        ego_proj_config = config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION
        ego_proj_config.local_map_shape = (2, map_size, map_size)
        ego_proj_config.map_scale = map_scale
        ego_proj_config.min_depth = min_depth
        ego_proj_config.max_depth = max_depth
        ego_proj_config.hfov = hfov
        ego_proj_config.vfov = vfov
        ego_proj_config.camera_height = camera_height
        ego_proj_config.height_thresholds = height_thresholds
        config.RL.ANS.OCCUPANCY_ANTICIPATOR.EGO_PROJECTION = ego_proj_config
        # Set the correct image scaling values
        config.RL.ANS.MAPPER.image_scale_hw = config.RL.ANS.image_scale_hw
        config.RL.ANS.LOCAL_POLICY.image_scale_hw = config.RL.ANS.image_scale_hw
        # Set the agent dynamics for the local policy
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.forward_step = (
            config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        config.RL.ANS.LOCAL_POLICY.AGENT_DYNAMICS.turn_angle = (
            config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        )
        config.freeze()

    def _setup_eval_config(self, checkpoint_config):
        config = self.config.clone()
        config.defrost()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            print("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _convert_actions_to_delta(self, actions):
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        delta_xyt = torch.zeros(1, 3, device=self.device)
        # Forward step
        act_mask = actions.squeeze(1) == 0
        delta_xyt[act_mask, 0] = sim_cfg.FORWARD_STEP_SIZE
        # Turn left
        act_mask = actions.squeeze(1) == 1
        delta_xyt[act_mask, 2] = math.radians(-sim_cfg.TURN_ANGLE)
        # Turn right
        act_mask = actions.squeeze(1) == 2
        delta_xyt[act_mask, 2] = math.radians(sim_cfg.TURN_ANGLE)
        return delta_xyt

    def _prepare_batch(self, observations, device=None, actions=None):
        imH, imW = self.config.RL.ANS.image_scale_hw
        device = self.device if device is None else device
        batch = batch_obs(observations, device=device)
        if batch["rgb"].size(1) != imH or batch["rgb"].size(2) != imW:
            rgb = rearrange(batch["rgb"], "b h w c -> b c h w")
            rgb = F.interpolate(rgb, (imH, imW), mode="bilinear")
            batch["rgb"] = rearrange(rgb, "b c h w -> b h w c")
        if batch["depth"].size(1) != imH or batch["depth"].size(2) != imW:
            depth = rearrange(batch["depth"], "b h w c -> b c h w")
            depth = F.interpolate(depth, (imH, imW), mode="nearest")
            batch["depth"] = rearrange(depth, "b c h w -> b h w c")
        # Compute ego_map_gt from depth
        ego_map_gt_b = self.depth_projection_net(
            rearrange(batch["depth"], "b h w c -> b c h w")
        )
        batch["ego_map_gt"] = rearrange(ego_map_gt_b, "b c h w -> b h w c")
        # Add previous action to batch as well
        batch["prev_actions"] = self.prev_actions
        # Add a rough pose estimate if GT pose is not available
        if "pose" not in batch:
            if self.prev_batch is None:
                # Set initial pose estimate to zero
                batch["pose"] = torch.zeros(1, 3).to(self.device)
            else:
                actions_delta = self._convert_actions_to_delta(self.prev_actions)
                batch["pose"] = add_pose(self.prev_batch["pose"], actions_delta)

        return batch

    def _remap_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Converts actions of exploration agent to actions for navigation.
        Remapping:
            0 -> 1 (forward)
            1 -> 2 (turn left)
            2 -> 3 (turn right)
            3 -> 0 (stop)
        """
        actions_rmp = torch.remainder(actions + 1, 4).long()
        return actions_rmp

    def reset(self):
        # Set models to evaluation
        self.mapper.eval()
        self.local_actor_critic.eval()

        # Reset agent states
        M = self.ans_cfg.overall_map_size
        self.state_estimates = {
            "pose_estimates": torch.zeros(1, 3).to(self.device),
            "map_states": torch.zeros(1, 2, M, M).to(self.device),
            "recurrent_hidden_states": torch.zeros(1, 1, self.ans_cfg.LOCAL_POLICY.hidden_size).to(self.device),
        }
        # Reset ANS states
        self.ans_net.reset()
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, device=self.device)
        self.prev_batch = None
        self.ep_time = torch.zeros(1, 1, device=self.device)
        self.ep_step = 0

    def act(self, observations):

        # ============================ Action step ============================
        batch = self._prepare_batch(observations)
        if self.prev_batch is None:
            self.prev_batch = copy.deepcopy(batch)

        prev_pose_estimates = self.state_estimates["pose_estimates"]
        with torch.no_grad():
            (
                _,
                _,
                mapper_outputs,
                local_policy_outputs,
                state_estimates,
            ) = self.ans_net.act(
                batch,
                self.prev_batch,
                self.state_estimates,
                self.ep_time,
                self.not_done_masks,
                deterministic=self.ans_cfg.LOCAL_POLICY.deterministic_flag,
            )
            actions = local_policy_outputs["actions"]
            # Make masks not done till reset (end of episode)
            self.not_done_masks = torch.ones(1, 1, device=self.device)
            self.prev_actions.copy_(actions)

        if self.ep_step == 0:
            state_estimates["pose_estimates"].copy_(prev_pose_estimates)

        self.ep_time += 1
        # Update prev batch
        for k, v in batch.items():
            self.prev_batch[k].copy_(v)

        # Remap actions from exploration to navigation agent.
        actions_rmp = self._remap_actions(actions)

        return {"action": [a[0].item() for a in actions_rmp][0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    parser.add_argument("--input-type", default="blind", choices=["blind", "rgb", "depth", "rgbd"])
    parser.add_argument("--exp-config", type=str, required=True)
    args = parser.parse_args()

    config = get_config(**vars(args))
    agent = OccAntAgent(config)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
