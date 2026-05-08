import json 
import os 
import math
import torch
import numpy as np
import pickle as pkl
import roboverse.constants as c
import copy
from tqdm import tqdm
from roboverse.unifiers.image_unifier import (image_unifier_transform,
                                              remove_keys)
from roboverse.datasets.lerobot.dataloader import le_sample_to_rv_sample
from roboverse.main import get_cfg
# define a dummy action for the simulator stability 
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def libero_to_lerobot_format(obs, language_instruction):
    # Convert LIBERO data to LEROBOT format

    # LEROBOT dataset format:
    # dict_keys(['image', 'wrist_image', 'state', 'actions', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'state_is_pad', 'actions_is_pad', 'image_is_pad', 'wrist_image_is_pad', 'task'])
    # specifically:
    # dataset state: torch.Size([1, 8])
    # dataset actions: torch.Size([9, 7])
    # dataset image: torch.Size([1, 3, 256, 256])

    # Convert images from (H, W, C) to (C, H, W) format
    image = torch.tensor(
        _img_to_numpy(obs["agentview_image"]) / 255.0, dtype=torch.float32
    ).permute(
        2, 0, 1
    )  # torch.Size([3, 256, 256])
    wrist_image = torch.tensor(
        _img_to_numpy(obs["robot0_eye_in_hand_image"]) / 255.0, dtype=torch.float32
    ).permute(
        2, 0, 1
    )  # torch.Size([3, 256, 256])

    # Construct state vector (robot pose + gripper state)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Create LEROBOT-compatible data structure
    lerobot_sample = {
        "image": image,  # torch.Size([1, 3, 256, 256])
        "wrist_image": wrist_image,  # torch.Size([1, 3, 256, 256])
        "state": state,  # torch.Size([1, 8])
        "task": language_instruction,  # Task description string
    }

    return lerobot_sample


def libero_to_rv_obs(obs, language_instruction, cfg):
    # TODO: we need to have ori_act in the obs. This is the last action by the model. Currently, we don't need it as no model is using it. But this might be needed in the future

    obs = libero_to_lerobot_format(obs, language_instruction)
    # Format LeRobot into RV Format, in numpy form.
    obs = le_sample_to_rv_sample(
        obs,
        history=cfg.history,
        horizon=cfg.horizon,
        **cfg.LEROBOT,
        add_ori_act=False,
        add_out_ori_act=False,  # always False as we don't know the ground truth action while evaluating
    )

    keys_to_remove = c.REQUIRED_KEYS_3D_COMPATIBLE
    if not cfg.IMAGE.return_ee:
        keys_to_remove.extend(c.REQUIRED_KEYS_EE_COMPATIBLE)
    if not cfg.IMAGE.return_ori_act:
        keys_to_remove.extend(c.REQUIRED_KEYS_ORIGINAL_ACTION)
    if not cfg.IMAGE.return_proprio:
        keys_to_remove.extend(c.REQUIRED_KEYS_PROPRIO)
    obs = remove_keys(obs, keys_to_remove)

    obs = image_unifier_transform(
        cfg, obs, sample_cam_list=cfg.IMAGE.cam_list, eval=True
    )

    # Final format on observations and send to tensor.
    for key in obs.keys():
        if key == "instr":
            obs["instr"] = [obs["instr"]]
        else:
            obs[key] = torch.tensor(obs[key][None], dtype=torch.float32).to(0)

    return obs

def _img_to_numpy(img):
    """
    image from the simulator is flipped. so we need to flip it back.
    """
    return np.ascontiguousarray(img[::-1, ::-1])


# for predicting the probabilistically scaled actions chunks 
def _predict_scaled_action_chunk(model, model_obs,inference_budget, temperature):
        """
        Internally implementing Probabilistic Inference Scaling.
        """
        particles = []
        log_probs = []

        # with torch.no_grad():
        for _ in range(inference_budget):
            out = model(**model_obs)
                
            # Extract chunk
            chunk = out["out_ori_act"][0].cpu().numpy()
            particles.append(chunk)
                
            # Extract log-likelihood if available, else use uniform (0.0)
            # This is where the 'Probabilistic' scaling happens
            lp = out.get("log_prob", torch.tensor(0.0)).item()
            log_probs.append(lp)

        particles = np.array(particles)
        log_probs = np.array(log_probs)

        # Softmax Weighting (Boltzmann distribution)
        shifted_probs = log_probs - np.max(log_probs)
        weights = np.exp(shifted_probs / temperature)
        weights /= np.sum(weights)

        # Expectation: Weighted average of particles
        return np.sum(particles * weights[:, np.newaxis, np.newaxis], axis=0)
 
 
# for handling the temporal ensemble logic separately 
# as well as implementing exponential decay ($weight^{dist}$) to give more importance to the most recent predictions.
# which was not there in the previous implementation where we simply took the average of the previous action chunks.

def _apply_temporal_aggregation(
    ensemble_prediction,
    action_horizon,
    ensemble_version,
    ensemble_2_weight,
    action_chunk,
    old_action_chunks,
    ):
        old_action_chunks.append(action_chunk)
        if len(old_action_chunks) > ensemble_prediction:
            old_action_chunks.pop(0)

        n_history = []
        combined = np.zeros_like(action_chunk)
        counts = np.zeros_like(action_chunk)

        for i, hist_chunk in enumerate(old_action_chunks[:-1]):
            if len(hist_chunk) <= action_horizon:
                continue
            
            shifted = hist_chunk[action_horizon:]
            n_history.append(shifted)
            
            weight = (ensemble_2_weight ** (len(old_action_chunks) - i - 1)) if ensemble_version == 2 else 0.5
            L = len(shifted)
            combined[:L] += weight * shifted
            counts[:L] += weight

        n_history.append(old_action_chunks[-1])
        combined += old_action_chunks[-1]
        counts += 1

        return (combined / counts), n_history


"""
Defining a class for action generation through MCTS in the action space 
"""
class ActionMCTS:
    def __init__(self, 
                 model,
                 env,
                 cfg,
                 n_simulations=10,
                 rollout_depth=2,
                 c_puct=1.4):
        
        self.model = model
        self.env = env
        self.cfg = cfg
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.c_puct = c_puct

    def _deep_reset_done(self, env, root_timestep):
        """
        Recursively finds and resets the '_done' flag in all environment wrappers.
        """
        """Resets flags AND restores the correct timestep."""
        if hasattr(env, "_done"): env._done = False
        if hasattr(env, "terminated"): env.terminated = False
        # Restore the 'real' timestep so the wrapper 
        # doesn't think the episode is over.
        if hasattr(env, "timestep"): env.timestep = root_timestep
        
        if hasattr(env, "env"):
            self._deep_reset_done(env.env, root_timestep)
    
    # disablign the dynamo compiler for this just to
    @torch._dynamo.disable
    def search(self, 
               initial_obs, 
               language_instruction):
        
        """
        function for performing MCTS search over action chunks.
        """
        # Save the current real state of the simulator to restore later
        # root_sim_state = self.env.get_sim_state()
        # Track the actual timestep from the real rollout
        root_timestep = getattr(self.env, "timestep", 0)
        root_sim_state = copy.deepcopy(self.env.sim.get_state())
        # In a real VLA-MCTS, we treat the model's chunk predictions as 'branches'
        # To keep it efficient, we sample N candidate chunks (particles) as our root actions
        candidates = []
        model_obs = libero_to_rv_obs(initial_obs, language_instruction, self.cfg)

        # setting the model to eval mode
        with torch.no_grad():
            for _ in range(self.n_simulations):
                out = self.model(**model_obs)
                chunk = out["out_ori_act"][0].cpu().numpy()
                log_p = out.get("log_prob", torch.tensor(0.0)).item()
                candidates.append({'chunk': chunk, 'score': 0, 'visits': 0, 'log_p': log_p})

        # Simulation Phase: Rollout each candidate in the 'imagined' simulator
        for cand in candidates:
            # Restore state for every simulation
            # We must tell the environment it's not 'done' before we start a new simulation
            # if hasattr(self.env, "_done"): self.env._done = False
            self._deep_reset_done(self.env,root_timestep)
            self.env.sim.set_state(root_sim_state)
            self.env.sim.forward()
            # Execute the chunk and see where we end up
            # We evaluate 'Value' based on change in reward or model-based heuristic
            accumulated_reward = 0
            current_obs = initial_obs
            """psecified the number of lookahead steps to 5 as the action horizon is 16 
            and we want to have a good balance between the depth of the search and the computational cost. 
            We can increase this number to have a deeper search but it will also increase the computational cost significantly."""
            for step in range(min(len(cand['chunk']), 5)): # Look ahead 5 steps
                _, reward, done, _ = self.env.step(cand['chunk'][step].tolist())
                accumulated_reward += reward
                if done: break
            
            # Value = Model Log-Prob (Policy) + Simulated Reward (Environment)
            cand['score'] = cand['log_p'] + (accumulated_reward * 10.0)
            cand['visits'] += 1

        # Restore the simulator to the exact state before we started 
        # self.env.set_sim_state(root_sim_state)
        self.env.sim.set_state(root_sim_state)
        self.env.sim.forward()
        # final reset the action pretending we never did the simulations
        # if hasattr(self.env, "_done"): self.env._done = False
        self._deep_reset_done(self.env,root_timestep)
        # Selection: Pick the best chunk based on MCTS scores
        best_cand = max(candidates, key=lambda x: x['score'])
        return best_cand['chunk']

class eval_run:
    """
    Args:
        env (_type_): _description_
        model (_type_): _description_
        cfg (_type_): _description_
        language_instruction (_type_): _description_
        init_state (_type_): _description_
        max_steps (_type_): _description_
        frame_skip (_type_): _description_
        action_horizon (_type_): _description_
        log_file_name (_type_): _description_
        save_all_data (_type_): _description_
    """
    def __init__(
        self,
        env,
        model,
        cfg,
        language_instruction,
        init_state,
        max_steps,
        frame_skip,
        action_horizon,
        log_file_name,
        save_all_data,
        ensemble_prediction,
        ensemble_version,
        ensemble_2_weight,
    ):
        # super().__init__(self)
        self.env = env 
        self.model = model
        self.cfg = cfg 
        self.language_instruction = language_instruction
        self.init_state = init_state  # the init state the simulation env is currently on 
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_horizon = action_horizon
        self.log_file_name = log_file_name
        self.save_all_data = save_all_data 
        self.ensemble_prediction = ensemble_prediction
        self.ensemble_version = ensemble_version
        self.ensemble_2_weight = ensemble_2_weight
        
    def get_final_actions_and_frame(self):
        # reset the env 
        self.env.reset()
        # defining action 
        action_i = 0
        # defining action chunk 
        action_chunk = None
        # getting the current obs to pass to the model 
        obs = self.env.set_init_state(self.init_state)
        # definig an array to store the frames 
        frames = []
        if self.save_all_data:
            all_actions = []
            all_obs = []

        if self.ensemble_prediction > 1:
            old_action_chunks = (
                []
            )  # maintian a list of action chunks from the previous steps

        t = 0
        
        # we based on the training define max steps > max_steps_used_during_training
        for t in tqdm(range(self.max_steps+self.frame_skip)):
            # frame skip defined for the simulator screen to settle down
            # as initial few frames will be very noisy can cause the simulator to go on a non recoverable path 
            
            if t<self.frame_skip : 
                # just take the obs dont feed it back in the simulator
                obs,reward,done,info = self.env.step(LIBERO_DUMMY_ACTION)
                t += 1
                if self.save_all_data:
                    all_actions.append(LIBERO_DUMMY_ACTION)
                    all_obs.append(obs)
                continue

            if action_i >= self.action_horizon or t == self.frame_skip:
                model_obs = libero_to_rv_obs(obs, self.language_instruction, self.cfg)
                out = self.model(**model_obs)
                action_chunk = out["out_ori_act"][0].numpy()  # 1 by 16 by 7
                if self.save_all_data:
                    all_actions.append(action_chunk)
                    all_obs.append(obs)

                if self.ensemble_prediction > 1:
                    old_action_chunks.append(action_chunk)
                    if len(old_action_chunks) > self.ensemble_prediction:
                        old_action_chunks.pop(0)

                    # updating the previous action chuncks
                    n_old_action_chunks = []
                    action_chunk = np.zeros_like(action_chunk)
                    action_chunk_count = np.zeros_like(action_chunk)
                    for i, _action_chunk in enumerate(old_action_chunks[:-1]):
                        # not added to n_old_action_chunks if the action chunk is shorter than the action horizon
                        if len(_action_chunk) <= self.action_horizon:
                            continue
                        else:
                            _action_chunk = _action_chunk[self.action_horizon:]
                            n_old_action_chunks.append(_action_chunk)

                        if self.ensemble_version == 1:
                            action_chunk[0 : len(_action_chunk)] += 0.5 * _action_chunk
                            action_chunk_count[0 : len(_action_chunk)] += 0.5
                        if self.ensemble_version == 2:
                            action_chunk[0 : len(_action_chunk)] += (
                                self.ensemble_2_weight ** (len(old_action_chunks) - i - 1)
                            ) * _action_chunk
                            action_chunk_count[
                                0 : len(_action_chunk)
                            ] += self.ensemble_2_weight ** (len(old_action_chunks) - i - 1)
                    #   adding the last action chunk
                    n_old_action_chunks.append(old_action_chunks[-1])
                    action_chunk += old_action_chunks[-1]
                    action_chunk_count += 1

                    old_action_chunks = n_old_action_chunks
                    action_chunk = action_chunk / action_chunk_count

                action_i = 0
                # Add a safeguard for the action horizon rollout.
                self.action_horizon = min(self.action_horizon, len(action_chunk))
            
            act = action_chunk[action_i]
            if not (act[-1] in [1, -1]):
                print(f"Action {act} is not in [1, -1]")
                if act[-1] > 0:
                    act[-1] = 1
                else:
                    act[-1] = -1
                    
            obs, reward, done, info = self.env.step(act.tolist())

            frames.append(_img_to_numpy(obs["agentview_image"]))
            if done:
                if self.save_all_data:
                    with open(self.log_file_name, "wb") as f:
                        pkl.dump(
                            {"actions": all_actions, "obs": all_obs},
                            f,
                        )
                return True, frames
            action_i += 1
            
        if self.save_all_data:
            with open(self.log_file_name, "wb") as f:
                pkl.dump(
                    {"actions": all_actions, "obs": all_obs},
                    f,
                )
        return False, frames
    
    # Probabilistic action sampling using particle based monte carlo method 
    
    def get_final_actions_and_frame_prob_pbmcm(
        self,
        inference_budget = 10 , #generally due to GPU limitations we 
        # can't have a very high inference budget. But even with 10 we can see a significant improvement in the performance of the model.
        # generally we may have 8 - 16 particles which is a sweet spot for the performance improvement and GPU limitations with H100.
        temperature = 0.05 
        # recommend starting with a temperature of 0.05. 
        # This is low enough to prioritize the model's best guess but high enough to allow the "Inference-Time Scaling" to actually smooth out the trajectory
        ):
        """
        Probabilistic Inference Scaling internally.
        """
        self.env.reset()
        action_i = 0
        action_chunk = None
        obs = self.env.set_init_state(self.init_state)
        frames = []
        
        if self.save_all_data:
            all_actions = []
            all_obs = []

        if self.ensemble_prediction > 1:
            old_action_chunks = []

        print(f"Running evaluation with inference_budget={inference_budget} and temperature={temperature}...")
        print("\n \033[32mRunning the simulation through the use of probabilistic scaling using the particle based monte carlo method for action sampling and selection.\033[0m \n")
        # Simulation Loop
        for t in tqdm(range(self.max_steps + self.frame_skip)):
            
            # 1. Warm-up phase
            if t < self.frame_skip:
                obs, reward, done, info = self.env.step(LIBERO_DUMMY_ACTION)
                if self.save_all_data:
                    all_actions.append(LIBERO_DUMMY_ACTION)
                    all_obs.append(obs)
                continue

            # 2. Inference phase (Triggered at horizon end or first step)
            if action_i >= self.action_horizon or t == self.frame_skip:
                model_obs = libero_to_rv_obs(obs, self.language_instruction, self.cfg)
                
                # Using Probabilistic Scaling
                # inference_budget = 1, the model acts normally (it takes its first guess). 
                # If we set inference_budget = 16, the model generates 16 different potential trajectories (particles)
                # it is like smaple data to generate which will be sampled from 
                action_chunk = _predict_scaled_action_chunk(
                    self.model, 
                    model_obs,
                    inference_budget, 
                    temperature
                )
                
                if self.save_all_data:
                    all_actions.append(action_chunk)
                    all_obs.append(obs)

                # 3. Temporal Ensemble Layer
                if self.ensemble_prediction > 1:
                    action_chunk, old_action_chunks = _apply_temporal_aggregation(
                        self.ensemble_prediction,
                        self.action_horizon,
                        self.ensemble_version,
                        self.ensemble_2_weight,
                        action_chunk, 
                        old_action_chunks,
                    )

                action_i = 0
                # Use current action_horizon as a safeguard
                current_horizon = min(self.action_horizon, len(action_chunk))

            # 4. Step Environment
            act = action_chunk[action_i]
            
            # Post-process gripper
            act[-1] = 1.0 if act[-1] > 0 else -1.0
                    
            obs, reward, done, info = self.env.step(act.tolist())
            frames.append(_img_to_numpy(obs["agentview_image"]))

            if done:
                if self.save_all_data:
                    self._dump_logs(all_actions, all_obs)
                return True, frames # SUCCESS
                
            action_i += 1
            
        if self.save_all_data:
            self._dump_logs(all_actions, all_obs)
            
        return False, frames # FAILURE in case of reaching max steps without success
    
    # Using the MCTS as action sampling and selection strategy in the action chunk selection 
    # while using the same temporal aggregation strategy as the previous implementation done above
    
    def get_final_actions_and_frame_mcts(
        self,
        n_simulations = 10,
        rollout_depth = 1,
        c_puct = 1.4
        ):
        """
        Modified function using MCTS for action selection before ensembling.
        """
        # Initialize the Engine
        planner = ActionMCTS(
            self.model, self.env, self.cfg, 
            n_simulations=n_simulations, 
            rollout_depth=rollout_depth,
            c_puct=c_puct
        )
    # This decorator stops the compiler from trying to optimize the 
    # # highly variable MCTS/Text-parsing logic
    #     with torch.compiler.disable():
        self.env.reset()
        action_i = 0
        action_chunk = None
        obs = self.env.set_init_state(self.init_state)
        frames = []
        
        if self.save_all_data:
            all_actions = []
            all_obs = []

        if self.ensemble_prediction > 1:
            old_action_chunks = []

        print(f"###################### Running MCTS evaluation with simulations={n_simulations} ######################")
        print("\n \033[34mRunning the simulation using MCTS 'Imagination' for action selection.\033[0m \n")

        for t in tqdm(range(self.max_steps + self.frame_skip)):
           
            if t < self.frame_skip:
                obs, reward, done, info = self.env.step(LIBERO_DUMMY_ACTION)
                if self.save_all_data:
                    all_actions.append(LIBERO_DUMMY_ACTION)
                    all_obs.append(obs)
                continue

            #Inference phase 
            if action_i >= self.action_horizon or t == self.frame_skip:
                
                # REPLACED: Using MCTS Planner to select the best chunk
                # It explores futures in the sim before returning the best 'chunk'
                action_chunk = planner.search(obs, self.language_instruction)
                
                if self.save_all_data:
                    all_actions.append(action_chunk)
                    all_obs.append(obs)

                # Temporal Ensemble Layer 
                if self.ensemble_prediction > 1:
                    action_chunk, old_action_chunks = _apply_temporal_aggregation(
                        self.ensemble_prediction,
                        self.action_horizon,
                        self.ensemble_version,
                        self.ensemble_2_weight,
                        action_chunk, 
                        old_action_chunks,
                    )

                action_i = 0
                self.action_horizon = min(self.action_horizon, len(action_chunk))

            # Step Environment 
            act = action_chunk[action_i]
            
            # Post-process gripper
            act[-1] = 1.0 if act[-1] > 0 else -1.0
                    
            obs, reward, done, info = self.env.step(act.tolist())
            frames.append(_img_to_numpy(obs["agentview_image"]))

            if done:
                if self.save_all_data:
                    self._dump_logs(all_actions, all_obs)
                return True, frames # SUCCESS
                
            action_i += 1
            
        if self.save_all_data:
            self._dump_logs(all_actions, all_obs)
            
        return False, frames