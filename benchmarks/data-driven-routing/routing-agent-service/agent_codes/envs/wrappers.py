# WANYU
# Example usage:
# in main()
# env = RealTimeWrapper(env, period_s=0.1)


import time
import gymnasium as gym

class RealTimeWrapper(gym.Wrapper):
    def __init__(self, env, period_s: float):
        super().__init__(env)
        self.period_s = float(period_s)
        self._next_tick = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        now = time.perf_counter()
        self._next_tick = now + self.period_s
        info = dict(info or {})
        info["period_s"] = self.period_s
        return obs, info

    def step(self, action):
        start = time.perf_counter()
        obs, r, term, trunc, info = self.env.step(action)
        now = time.perf_counter()
        sleep_for = self._next_tick - now
        if sleep_for > 0:
            time.sleep(sleep_for)
            jitter = 0.0
            self._next_tick += self.period_s
        else:
            jitter = -sleep_for
            missed = int((-sleep_for) // self.period_s) + 1
            self._next_tick += missed * self.period_s
        info = dict(info or {})
        info.update({"step_time_s": now - start, "neg_slack_s": jitter})
        return obs, r, term, trunc, info


