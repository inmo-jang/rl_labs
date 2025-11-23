from gym.envs.registration import register
from gym import logger, spaces
import gym
import pygame
import numpy as np
from typing import Optional
from gym.error import DependencyNotInstalled

class CartPoleSerialDoublePendulumEnv(gym.Env[np.ndarray, int]):
    """
    Cart + 2-link serial inverted pendulum (education-grade dynamics).
    State: [x, xdot, th1, th1dot, th2, th2dot]
      - th1: 1번 링크의 절대각(수직 위에서 시계/반시계, 아래에서 시작하려면 pi 부근)
      - th2: 2번 링크의 '상대각'이 아니라 '절대각'으로 두었습니다. (렌더/직관 용이)
        * relative를 원하면 코드의 cos/sin에서 th12 처리를 바꾸면 됩니다.
    Action: {0:left, 1:right} -> force +/- self.force_mag on cart
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        # --- physical params (CartPole와 유사 스케일) ---
        self.g = 9.8
        self.m0 = 1.0      # cart
        self.m1 = 0.1      # link1
        self.m2 = 0.1      # link2
        self.l1 = 0.25      # half-length of link1 (COM까지 거리)
        self.l2 = 0.25      # half-length of link2 (COM까지 거리)
        # moment of inertia about joint (rod about one end): I = m*(2l)^2/3 = 4/3*m*l^2
        self.I1 = 4.0/3.0 * self.m1 * self.l1**2
        self.I2 = 4.0/3.0 * self.m2 * self.l2**2

        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        # termination bounds (위치는 CartPole처럼, 각도 제한은 넉넉하게)
        self.x_threshold = 2.4
        self.theta_threshold_radians = float("inf")

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # render
        self.render_mode = render_mode
        self.screen_width = 900
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.surf = None
        self.isopen = True

        self.state = None
        self.steps_beyond_terminated = None

    @staticmethod
    def _wrap(a: float) -> float:
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        noise = 0.01
        # 아래 방향(pi)에서 약간 노이즈
        x = self.np_random.uniform(low=-0.05, high=0.05)
        xdot = 0.0
        th1 = np.pi + self.np_random.uniform(-noise, noise)
        th1dot = 0.0
        th2 = np.pi + self.np_random.uniform(-noise, noise)
        th2dot = 0.0
        self.state = (x, xdot, th1, th1dot, th2, th2dot)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    # --- M(q) qdd + C(q,qd) + G(q) = B u  (간단화 버전) ---
    def _dyn_rhs(self, u, x, xdot, th1, th1dot, th2, th2dot):
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c12 = np.cos(th1 - th2)   # th2를 절대각으로 뒀기 때문에 상호항은 (th1 - th2)
        s12 = np.sin(th1 - th2)

        m0, m1, m2 = self.m0, self.m1, self.m2
        l1, l2 = self.l1, self.l2
        I1, I2 = self.I1, self.I2
        g = self.g

        # ---- Mass matrix M (대칭) ----
        # 카트 + 두 링크 직렬의 표준형을 교육용으로 단순화
        # 참고: 링크2 COM과 링크1 끝점 간의 상호항 포함
        M11 = (m0 + m1 + m2)
        M12 =  (m1*l1 + m2*2*l1) * c1 + m2*l2 * c2
        M13 =  m2*l2 * c2
        M22 =  I1 + m1*l1**2 + I2 + m2*( (2*l1)**2 + l2**2 + 2*(2*l1)*l2*c12 )
        M23 =  I2 + m2*( l2**2 + (2*l1)*l2*c12 )
        M33 =  I2 + m2*l2**2

        M = np.array([
            [M11, M12, M13],
            [M12, M22, M23],
            [M13, M23, M33],
        ], dtype=np.float64)

        # ---- Coriolis/Centrifugal C (벡터) ----
        # 교육용 근사: 주요 상호항만 유지
        # x항에는 각속도와 결합된 작은 항들을 생략(안정·단순화 목적)
        h = m2*(2*l1)*l2*s12  # 링크1–링크2 상호항
        C1 = 0.0
        C2 = - h * (2*th1dot*th2dot + th2dot**2)
        C3 =  + h * (th1dot**2)

        C = np.array([C1, C2, C3], dtype=np.float64)

        # ---- Gravity G (벡터) ----
        # x에는 중력 항 없음
        G1 = 0.0
        # 절대각 기준: 아래가 +π 이므로 sin(th) 부호 주의 (수직 위=0일 때 기준과 동일식 사용)
        G2 = - (m1*g*l1*np.sin(th1) + m2*g*(2*l1*np.sin(th1) + l2*np.sin(th2)))
        G3 = - (m2*g*l2*np.sin(th2))
        G = np.array([G1, G2, G3], dtype=np.float64)

        # 입력 매핑 B u (u는 카트 수평힘)
        B = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # qdd = M^{-1} (B u - C - G)
        rhs = B*u - C - G
        qdd = np.linalg.solve(M, rhs)
        xdd, th1dd, th2dd = qdd.tolist()
        return xdd, th1dd, th2dd

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action"
        x, xdot, th1, th1dot, th2, th2dot = self.state
        u = self.force_mag if action == 1 else -self.force_mag

        xdd, th1dd, th2dd = self._dyn_rhs(u, x, xdot, th1, th1dot, th2, th2dot)

        if self.kinematics_integrator == "euler":
            x      = x      + self.tau * xdot
            xdot   = xdot   + self.tau * xdd
            th1    = th1    + self.tau * th1dot
            th1dot = th1dot + self.tau * th1dd
            th2    = th2    + self.tau * th2dot
            th2dot = th2dot + self.tau * th2dd
        else:
            # semi-implicit euler
            xdot   = xdot   + self.tau * xdd
            x      = x      + self.tau * xdot
            th1dot = th1dot + self.tau * th1dd
            th1    = th1    + self.tau * th1dot
            th2dot = th2dot + self.tau * th2dd
            th2    = th2    + self.tau * th2dot

        th1 = self._wrap(th1)
        th2 = self._wrap(th2)
        self.state = (x, xdot, th1, th1dot, th2, th2dot)

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        reward = 1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("Render mode not set.")
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled("pygame not installed.")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        cartwidth, cartheight = 50.0, 30.0
        polewidth = 8.0

        x, _, th1, _, th2, _ = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # cart
        l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        axleoffset = cartheight / 4.0
        cartx = x * scale + self.screen_width / 2.0
        carty = 100
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(cx + cartx, cy + carty) for (cx, cy) in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # link1 polygon (길이 = 2*l1 * scale)
        L1 = 2*self.l1*scale
        l1_, r1_, t1_, b1_ = -polewidth/2, polewidth/2, L1 - polewidth/2, -polewidth/2
        link1 = []
        for coord in [(l1_, b1_), (l1_, t1_), (r1_, t1_), (r1_, b1_)]:
            v = pygame.math.Vector2(coord).rotate_rad(-th1)
            link1.append((v[0] + cartx, v[1] + carty + axleoffset))
        gfxdraw.aapolygon(self.surf, link1, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, link1, (202, 152, 101))

        # joint1 (cart axle)
        gfxdraw.aacircle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth/2), (129,132,203))
        gfxdraw.filled_circle(self.surf, int(cartx), int(carty + axleoffset), int(polewidth/2), (129,132,203))

        # link1 tip (joint2 위치) 좌표
        tip1_vec = pygame.math.Vector2(0, L1).rotate_rad(-th1)
        j2x = cartx + tip1_vec[0]
        j2y = carty + axleoffset + tip1_vec[1]

        # link2 polygon (joint2에서 시작, 절대각 th2)
        L2 = 2*self.l2*scale
        l2_, r2_, t2_, b2_ = -polewidth/2, polewidth/2, L2 - polewidth/2, -polewidth/2
        link2 = []
        for coord in [(l2_, b2_), (l2_, t2_), (r2_, t2_), (r2_, b2_)]:
            v = pygame.math.Vector2(coord).rotate_rad(-th2)
            link2.append((v[0] + j2x, v[1] + j2y))
        gfxdraw.aapolygon(self.surf, link2, (180, 120, 80))
        gfxdraw.filled_polygon(self.surf, link2, (180, 120, 80))

        # joint2
        gfxdraw.aacircle(self.surf, int(j2x), int(j2y), int(polewidth/2), (90, 90, 180))
        gfxdraw.filled_circle(self.surf, int(j2x), int(j2y), int(polewidth/2), (90, 90, 180))

        # ground line
        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1,0,2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False



class SerialDoublePendulumGym:
    def __init__(self, render_mode=None, episode_length=1000):
        register(
            id='CartPoleSerialDoublePendulumEnv',
            entry_point='__main__:CartPoleSerialDoublePendulumEnv',
        )
        self.env = gym.make('CartPoleSerialDoublePendulumEnv', render_mode=render_mode)
        self.env.reset()
        self.render_mode = render_mode
        self.metadata = self.env.metadata
        self.steps = 0
        self.episode_length = episode_length

    def get_env(self): return self.env
    def render(self):  return self.env.render()
    def close(self):   self.env.close()

    def reset(self, seed=None, options=None):
        self.steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        s, r, term, trunc, info = self.env.step(action)
        # (옵션) 각도 wrap 재적용
        s[2] = float(np.arctan2(np.sin(s[2]), np.cos(s[2])))
        s[4] = float(np.arctan2(np.sin(s[4]), np.cos(s[4])))
        self.steps += 1
        if self.steps > self.episode_length:
            trunc = True
        return s, r, term, trunc, info

    def get_action_space(self):       return self.env.action_space
    def get_observation_space(self):  return self.env.observation_space

    def key_action(self, previous_action=0):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  return 0
        if keys[pygame.K_RIGHT]: return 1
        if keys[pygame.K_r]:     self.reset()
        return previous_action

    def handle_keyboard_input(self):
        clock = pygame.time.Clock()
        terminated = False
        previous_action = 0
        while not terminated:
            self.env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            action = self.key_action(previous_action)
            s, r, terminated, truncated, _ = self.step(action)
            print(f"reward={r:.1f}, x={s[0]:.2f}, xdot={s[1]:.2f}, th1={s[2]:.2f}, th1dot={s[3]:.2f}, th2={s[4]:.2f}, th2dot={s[5]:.2f}")
            pygame.display.flip()
            clock.tick(50)
            previous_action = action
        pygame.quit()
        self.env.close()





if __name__ == "__main__":
    env = SerialDoublePendulumGym(render_mode="human")
    env.reset()
    env.handle_keyboard_input()
