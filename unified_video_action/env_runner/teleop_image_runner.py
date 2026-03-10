import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from unified_video_action.env.teleop.teleop_image_env import TeleopImageEnv
from unified_video_action.gym_util.async_vector_env import AsyncVectorEnv
from unified_video_action.gym_util.multistep_wrapper import MultiStepWrapper
from unified_video_action.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.env_runner.base_image_runner import BaseImageRunner


class TeleopImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        dataset_path,
        arm_hand_name,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=300,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=96,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    TeleopImageEnv(
                        dataset_path=dataset_path,
                        arm_hand_name=arm_hand_name,
                        render_size=render_size,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, shared_memory=False)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy, **kwargs):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])
            obs = env.reset()
            past_action_list = []
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval TeleopImageRunner {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                np_obs_dict = dict(obs)
                if self.past_action and len(past_action_list) > 1:
                    np_obs_dict["past_action"] = np.concatenate(past_action_list, axis=1)

                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, **kwargs)

                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                action = np_action_dict["action"]
                obs, reward, done, info = env.step(action)
                done = np.all(done)

                past_action_list.append(action)
                if len(past_action_list) > 2:
                    past_action_list.pop(0)
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[this_local_slice]
        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f"sim_video_{seed}"] = wandb.Video(video_path)

        for prefix, value in max_rewards.items():
            log_data[prefix + "mean_score"] = np.mean(value)

        return log_data
