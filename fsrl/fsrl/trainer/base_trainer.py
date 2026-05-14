import json
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm
from tianshou.utils import DummyTqdm, MovAvg, deprecation, tqdm_config

from fsrl.data import BasicCollector, FastCollector
from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger


COLLECTOR_OPTIONAL_LOG_KEYS = (
    "success_rate",
    "safe_success_rate",
    "crash_free_rate",
    "no_out_of_road_rate",
    "success_rate_window100",
    "safe_success_rate_window100",
    "crash_free_rate_window100",
    "no_out_of_road_rate_window100",
    "avg_speed_km_h",
    "event_cost",
    "risk_field_cost",
    "risk_field_road_cost",
    "risk_field_vehicle_cost",
    "risk_field_object_cost",
    "base_reward",
    "driving_component_reward",
    "speed_component_reward",
    "normalized_speed_reward",
    "longitudinal_progress",
    "positive_road",
    "reward_override_active",
    "reward_override_delta",
    "final_reward",
    "smoothness_penalty",
    "smoothness_penalty_raw",
    "action_smoothness_penalty",
    "action_smoothness_penalty_raw",
    "steering_delta",
    "steering_switch",
    "steering_switch_count",
    "steering_absolute_value",
    "steering_jerk",
    "steering_smoothness_penalty",
    "steering_switch_penalty",
    "steering_switch_count_penalty",
    "steering_absolute_penalty",
    "steering_penalty",
    "steering_jerk_smoothness_penalty",
    "throttle_smoothness_penalty",
    "jerk_smoothness_penalty",
    "lateral_now",
    "lateral_norm",
    "lateral_score",
    "lateral_speed_gate",
    "lateral_forward_gate",
    "lateral_broken_line_gate",
    "lateral_reward",
    "lateral_velocity_m_s",
    "lateral_acceleration_m_s2",
    "lateral_jerk_m_s3",
    "yaw_rate_rad_s",
    "yaw_acceleration_rad_s2",
    "base_reward_step_mean",
    "driving_component_reward_step_mean",
    "speed_component_reward_step_mean",
    "normalized_speed_reward_step_mean",
    "longitudinal_progress_step_mean",
    "positive_road_step_mean",
    "reward_override_active_step_mean",
    "reward_override_delta_step_mean",
    "final_reward_step_mean",
    "smoothness_penalty_step_mean",
    "smoothness_penalty_raw_step_mean",
    "action_smoothness_penalty_step_mean",
    "action_smoothness_penalty_raw_step_mean",
    "steering_delta_step_mean",
    "steering_switch_step_mean",
    "steering_switch_count_step_mean",
    "steering_absolute_value_step_mean",
    "steering_jerk_step_mean",
    "steering_smoothness_penalty_step_mean",
    "steering_switch_penalty_step_mean",
    "steering_switch_count_penalty_step_mean",
    "steering_absolute_penalty_step_mean",
    "steering_penalty_step_mean",
    "steering_jerk_smoothness_penalty_step_mean",
    "throttle_smoothness_penalty_step_mean",
    "jerk_smoothness_penalty_step_mean",
    "lateral_now_step_mean",
    "lateral_norm_step_mean",
    "lateral_score_step_mean",
    "lateral_speed_gate_step_mean",
    "lateral_forward_gate_step_mean",
    "lateral_broken_line_gate_step_mean",
    "lateral_reward_step_mean",
    "risk_field_cost_step_mean",
    "risk_field_road_cost_step_mean",
    "risk_field_vehicle_cost_step_mean",
    "risk_field_object_cost_step_mean",
    "lateral_velocity_m_s_step_mean",
    "lateral_acceleration_m_s2_step_mean",
    "lateral_jerk_m_s3_step_mean",
    "yaw_rate_rad_s_step_mean",
    "yaw_acceleration_rad_s2_step_mean",
)


class BaseTrainer(ABC):
    """An iterator base class for trainers procedure.

    Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results on
    every epoch. The usage of the trainer is almost identical with tianshou's trainer,
    but with some modifications of the parameters.

    :param learning_type str: type of learning iterator, available choices are
        "offpolicy" and "onpolicy", we don't support "offline" yet.
    :param policy: an instance of the :class:`~fsrl.policy.BasePolicy` class.
    :param train_collector: the collector used for training.
    :param test_collector: the collector used for testing. If it's None, then no testing
        will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training process
        might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int cost_limit: the constraint violation threshold.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning, for
        example, set it to 2 means the policy needs to learn each given batch data twice.
        (on-policy method)
    :param float update_per_step: the number of gradient steps per env_step (off-policy).
    :param float save_model_interval: how many epochs to save one checkpoint.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatedly in each epoch.
    :param function stop_fn: a function with signature ``f(reward, cost) ->
        bool``, receives the average undiscounted returns of the testing result, returns
        a boolean which indicates whether reaching the goal.
    :param bool resume_from_log: resume env_step and other metadata from existing
        tensorboard log. Default to False.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print tabular information. Default to True.
    :param bool show_progress: whether to display a progress bar when training. Default
        to True.
    """

    @staticmethod
    def gen_doc(learning_type: str) -> str:
        """Document string for subclass trainer."""
        step_means = f'The "step" in {learning_type} trainer means '
        if learning_type != "offline":
            step_means += "an environment step (a.k.a. transition)."
        else:  # offline
            step_means += "a gradient step."

        trainer_name = learning_type.capitalize() + "Trainer"

        return f"""An iterator class for {learning_type} trainer procedure.

        Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results
        on every epoch.

        {step_means}

        Example usage:

        ::

            trainer = {trainer_name}(...) for epoch, epoch_stat, info in trainer:
                print("Epoch:", epoch) print(epoch_stat) print(info)
                do_something_with_policy() query_something_about_policy()
                make_a_plot_with(epoch_stat) display(info)

        - epoch int: the epoch number
        - epoch_stat dict: a large collection of metrics of the current epoch
        - info dict: result returned from :func:`~fsrl.BaseTrainer.gather_update_info`

        You can even iterate on several trainers at the same time:

        ::

            trainer1 = {trainer_name}(...) trainer2 = {trainer_name}(...) for result1,
            result2, ... in zip(trainer1, trainer2, ...):
                compare_results(result1, result2, ...)
        """

    def __init__(
        self,
        learning_type: str,
        policy: BasePolicy,
        train_collector: FastCollector,
        test_collector: Optional[FastCollector] = None,
        max_epoch: int = 100,
        batch_size: int = 512,
        cost_limit: float = np.inf,
        step_per_epoch: Optional[int] = None,
        repeat_per_collect: Optional[int] = None,
        update_per_step: Union[int, float] = 1,
        save_model_interval: int = 1,
        episode_per_test: Optional[int] = None,
        episode_per_collect: int = 1,
        test_every_episode: int = 0,
        save_test_artifacts: bool = False,
        artifact_env_factory: Optional[Callable[[], Any]] = None,
        stop_fn: Optional[Callable[[float, float], bool]] = None,
        resume_from_log: bool = False,
        logger: BaseLogger = BaseLogger(),
        verbose: bool = True,
        show_progress: bool = True,
    ):

        self.policy = policy
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.logger = logger
        self.cost_limit = cost_limit

        self.start_time = time.time()

        # used for determining stopping criterio.
        self.stats_smoother: DefaultDict[str, MovAvg] = defaultdict(MovAvg)
        # The best performance is dertemined by [reward, feasibility status]. If two
        # policies are both (in)feasible, the higher reward one is better. Otherwise, the
        # feasible one is better.
        self.best_perf_rew = -np.inf
        self.best_perf_cost = np.inf
        self.start_epoch = 0
        self.env_step = 0
        self.cum_cost = 0
        self.cum_episode = 0
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.episode_per_collect = episode_per_collect
        self.episode_per_test = episode_per_test
        self.test_every_episode = max(int(test_every_episode), 0)
        self.save_test_artifacts = save_test_artifacts
        self.artifact_env_factory = artifact_env_factory

        self.update_per_step = update_per_step
        self.save_model_interval = save_model_interval
        self.repeat_per_collect = repeat_per_collect

        self.batch_size = batch_size

        self.stop_fn = stop_fn

        self.verbose = verbose
        self.show_progress = show_progress
        self.resume_from_log = resume_from_log

        self.epoch = self.start_epoch
        self.best_epoch = self.start_epoch
        self.stop_fn_flag = False
        self.next_test_episode = self.test_every_episode if self.test_every_episode > 0 else None
        self.latest_test_reward: Optional[float] = None
        self.latest_test_cost: Optional[float] = None

    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.env_step = 0
        # TODO
        # if self.resume_from_log:
        #     self.start_epoch, self.env_step = \
        #         self.logger.restore_data()
        #     self.best_epoch = self.start_epoch

        self.start_time = time.time()

        self.train_collector.reset_stat()

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            self.test_collector.reset_stat()

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.next_test_episode = self.test_every_episode if self.test_every_episode > 0 else None
        self.latest_test_reward = None
        self.latest_test_cost = None

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> Tuple[int, Dict, Dict]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        # iterator exhaustion check
        if self.epoch > self.max_epoch:
            raise StopIteration

        # exit flag 1, when stop_fn succeeds in train_step or test_step
        if self.stop_fn_flag:
            raise StopIteration

        # set policy in train mode (not training the policy)
        self.policy.train()

        progress = tqdm.tqdm if self.show_progress else DummyTqdm
        # perform n step_per_epoch
        with progress(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:

                stats_train = self.train_step()
                t.update(stats_train["n/st"])

                self.policy_update_fn(stats_train)
                self._maybe_run_periodic_test()

                t.set_postfix(
                    cost=stats_train["cost"],
                    rew=stats_train["rew"],
                    length=stats_train["len"]
                )

                self.logger.write_without_reset(self.env_step)
                if self.stop_fn_flag:
                    break

        # test
        if self.test_collector is not None and self.test_every_episode <= 0:
            self.test_step()

        # train and test collector info
        update_info = self.gather_update_info()
        self.logger.store(tab="update", **update_info)

        if self.epoch % self.save_model_interval == 0:
            self.logger.save_checkpoint()

        if self.test_every_episode <= 0 and self.perf_is_better(test=True):
            self.logger.save_checkpoint(suffix="best")

        if self.stop_fn and self.stop_fn(self.best_perf_rew, self.best_perf_cost):
            self.stop_fn_flag = True
            self.logger.print("Early stop due to the stop_fn met.", "red")

        epoch_stats = self.logger.stats_mean

        # after write, all the stats will be resetted.
        self.logger.write(self.env_step, display=self.verbose)

        update_info.update(
            {
                "best_reward": self.best_perf_rew,
                "best_cost": self.best_perf_cost
            }
        )

        return self.epoch, epoch_stats, update_info

    def perf_is_better(self, test: bool = True) -> bool:
        # use the testing or the training metric to determine the best
        mode = "test" if test and self.test_collector is not None else "train"
        rew = self.logger.get_mean(mode + "/reward")
        cost = self.logger.get_mean(mode + "/cost")
        return self._update_best_performance(rew, cost)

    def _update_best_performance(self, rew: float, cost: float) -> bool:
        if self.best_perf_cost > self.cost_limit:
            if cost <= self.cost_limit or rew > self.best_perf_rew:
                self.best_perf_cost = cost
                self.best_perf_rew = rew
                return True
        else:
            if cost <= self.cost_limit and rew > self.best_perf_rew:
                self.best_perf_cost = cost
                self.best_perf_rew = rew
                return True
        return False

    def _maybe_run_periodic_test(self) -> None:
        if self.test_collector is None or self.test_every_episode <= 0:
            return
        while self.next_test_episode is not None and self.cum_episode >= self.next_test_episode:
            trigger_episode = self.next_test_episode
            self.next_test_episode += self.test_every_episode
            self._run_periodic_test(trigger_episode)
            if self.stop_fn_flag:
                break

    def _run_periodic_test(self, trigger_episode: int) -> None:
        stats_test = self.test_step()
        self.latest_test_reward = float(stats_test["rew"])
        self.latest_test_cost = float(stats_test["cost"])
        self.logger.store(tab="update", test_trigger_episode=trigger_episode)

        if self._update_best_performance(self.latest_test_reward, self.latest_test_cost):
            self.logger.save_checkpoint(suffix="best")

        if self.save_test_artifacts:
            self._save_test_artifacts(trigger_episode, stats_test)

        if self.stop_fn and self.stop_fn(self.best_perf_rew, self.best_perf_cost):
            self.stop_fn_flag = True
            self.logger.print("Early stop due to the stop_fn met.", "red")

        self.logger.write_without_reset(self.env_step)
        self.policy.train()

    def _save_test_artifacts(self, trigger_episode: int, stats_test: Dict[str, Any]) -> None:
        if not self.logger.log_dir:
            return

        artifact_dir = os.path.join(
            self.logger.log_dir,
            "periodic_eval",
            self._build_artifact_dir_name(trigger_episode),
        )
        artifact_dir = self._ensure_unique_dir(artifact_dir)
        checkpoint_dir = os.path.join(artifact_dir, "checkpoint")
        render_dir = os.path.join(artifact_dir, "render")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)

        torch.save({"model": self.policy.state_dict()}, os.path.join(checkpoint_dir, "model.pt"))

        metrics = {
            "epoch": self.epoch,
            "env_step": int(self.env_step),
            "train_episode": int(self.cum_episode),
            "test_trigger_episode": int(trigger_episode),
            "test_stats": self._to_serializable_dict(stats_test),
        }

        if self.artifact_env_factory is not None:
            bev_gif_path = os.path.join(render_dir, "bev.gif")
            front_gif_path = os.path.join(render_dir, "front.gif")
            try:
                metrics["debug_rollout"] = self._generate_debug_gifs(
                    bev_gif_path,
                    front_gif_path,
                )
            except Exception as exc:  # pragma: no cover
                metrics["debug_rollout_error"] = str(exc)

        with open(os.path.join(artifact_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_front_gif_frame(env: Any, gif_writer: Any) -> None:
        if getattr(env, "main_camera", None) is None:
            raise RuntimeError(
                "MetaDrive前视摄像头不可用。训练期保存前视GIF时需要启用环境 use_render=True。"
            )
        # MetaDrive main_camera.perceive 返回 BGR，这里转成 imageio 需要的 RGB。
        frame = env.main_camera.perceive(to_float=False)[..., ::-1]
        gif_writer.append_data(frame)

    def _generate_debug_gifs(self, bev_gif_path: str, front_gif_path: str) -> Dict[str, Any]:
        env = self.artifact_env_factory()
        render_env = getattr(env, "unwrapped", env)
        front_gif_writer = None

        try:
            collector = BasicCollector(self.policy, env)
            self.policy.eval()

            # TopDownRenderer is created lazily and cleared on the next reset, so we
            # initialize it once up front and keep the same episode alive until the
            # GIF is written.
            render_env.render(
                mode="topdown",
                window=False,
                screen_record=True,
                target_agent_heading_up=True,
                film_size=(2000, 2000),
                screen_size=(800, 800),
            )

            front_gif_enabled = getattr(render_env, "main_camera", None) is not None
            front_gif_error = None
            if front_gif_enabled:
                front_gif_writer = imageio.get_writer(
                    front_gif_path,
                    mode="I",
                    duration=1.0 / 10.0,
                    loop=0,
                )
                self._append_front_gif_frame(render_env, front_gif_writer)
            else:
                front_gif_error = (
                    "MetaDrive前视摄像头不可用。训练期保存前视GIF时需要启用环境 use_render=True。"
                )

            rollout_stats = collector.collect(
                n_episode=1,
                render=True,
                render_mode="topdown",
                render_kwargs={
                    "window": False,
                    "screen_record": True,
                    "target_agent_heading_up": True,
                },
                step_callback=(
                    None
                    if front_gif_writer is None else
                    lambda collector_env, _: self._append_front_gif_frame(
                        getattr(collector_env, "unwrapped", collector_env),
                        front_gif_writer,
                    )
                ),
                reset_after_collect=False,
            )

            if render_env.top_down_renderer is None:
                raise RuntimeError("TopDownRenderer未初始化。请确保环境配置支持渲染。")

            if not render_env.top_down_renderer.screen_frames:
                raise RuntimeError(
                    f"未记录到任何BEV帧。已收集{rollout_stats.get('n/st', 0)}步。"
                    f"请检查环境的渲染配置是否正确。"
                )

            render_env.top_down_renderer.generate_gif(bev_gif_path)

            result = self._to_serializable_dict(rollout_stats)
            result["bev_gif_path"] = bev_gif_path
            result["front_gif_path"] = front_gif_path if front_gif_writer is not None else None
            if front_gif_error is not None:
                result["front_gif_error"] = front_gif_error
            return result
        finally:
            if front_gif_writer is not None:
                front_gif_writer.close()
            env.close()

    @staticmethod
    def _to_serializable_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result

    def _build_artifact_dir_name(self, trigger_episode: int) -> str:
        return f"episode_{trigger_episode:06d}_step_{int(self.env_step):09d}"

    @staticmethod
    def _ensure_unique_dir(path: str) -> str:
        if not os.path.exists(path):
            return path
        suffix = 1
        while True:
            candidate = f"{path}_{suffix:02d}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    def _store_collect_stats(self, prefix: str, stats: Dict[str, Any]) -> None:
        for key in COLLECTOR_OPTIONAL_LOG_KEYS:
            if key in stats:
                self.logger.store(**{f"{prefix}/{key}": stats[key]})

    def test_step(self) -> Dict[str, Any]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        self.test_collector.reset_env()
        self.test_collector.reset_buffer()
        self.policy.eval()
        stats_test = self.test_collector.collect(n_episode=self.episode_per_test)

        self.logger.store(
            **{
                "test/reward": stats_test["rew"],
                "test/cost": stats_test["cost"],
                "test/length": int(stats_test["len"]),
            }
        )
        self._store_collect_stats("test", stats_test)
        return stats_test

    def train_step(self) -> Dict[str, Any]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        stats_train = self.train_collector.collect(self.episode_per_collect)

        self.env_step += int(stats_train["n/st"])
        self.cum_cost += stats_train["total_cost"]
        self.cum_episode += int(stats_train["n/ep"])
        self.logger.store(
            **{
                "update/episode": self.cum_episode,
                "update/cum_cost": self.cum_cost,
                "train/reward": stats_train["rew"],
                "train/cost": stats_train["cost"],
                "train/length": int(stats_train["len"]),
            }
        )
        self._store_collect_stats("train", stats_train)
        return stats_train

    @abstractmethod
    def policy_update_fn(self, result: Dict[str, Any]) -> None:
        """Policy update function for different trainer implementation.

        :param result: collector's return value.
        """

    def run(self) -> Dict[str, Union[float, str]]:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed (feed
        the entire iterator into a zero-length deque).
        """
        deque(self, maxlen=0)
        return self.gather_update_info()

    def gather_update_info(self) -> Dict[str, Any]:
        """A simple wrapper of gathering information from collectors.

        :return: A dictionary with the following keys:

            * ``train_collector_time`` the time (s) for collecting transitions in the \
                training collector;
            * ``train_model_time`` the time (s) for training models;
            * ``train_speed`` the speed of training (env_step per second);
            * ``test_time`` the time (s) for testing;
            * ``test_speed`` the speed of testing (env_step per second);
            * ``duration`` the total elapsed time (s).
        """
        duration = max(0, time.time() - self.start_time)
        model_time = max(0, duration - self.train_collector.collect_time)
        result = {"duration": duration}
        if self.test_collector is not None:
            collect_test = self.test_collector.collect_time
            model_time = max(0, model_time - collect_test)
            test_speed = self.test_collector.collect_step / collect_test if collect_test > 0 else 0.0
            result.update(
                {
                    "test_time": collect_test,
                    "test_speed": test_speed,
                    "duration": duration,
                }
            )
            train_duration = max(duration - collect_test, 1e-9)
            train_speed = self.train_collector.collect_step / train_duration
        else:
            train_speed = self.train_collector.collect_step / max(duration, 1e-9)
        result.update(
            {
                "train_collector_time": self.train_collector.collect_time,
                "train_model_time": model_time,
                "train_speed": train_speed,
                "remaining_epoch": self.max_epoch - self.epoch
            }
        )
        return result
