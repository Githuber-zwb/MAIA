import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete

# update bounds to center around agent
cam_range = 2

class IAMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True):

        self.world = world
        self.world_length = self.world.episode_length
        self.current_step = 0
        self.agents = self.world.transporters
        # set required vectorized gym env property
        self.n = self.world.num_transporter
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.max_step = 10000

        self.post_step_callback = post_step_callback

        # environment parameters

        # if true, every agent has the same reward
        self.shared_reward = world.shared_reward
        #self.shared_reward = True
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            self.action_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(2,), dtype=np.float32))  # [-inf,inf]

            obs_dim = len(observation_callback(agent, self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.transporters
        # set action for each agent
        # action_n:a list, contains num_agent elements,each element is a (single_action_dim,)shape array. 
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()  # core.step()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {}
            # info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)
        global_reward = self._get_global_reward()

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n) + global_reward
        if self.shared_reward:
            reward_n = [[reward]] * self.n
            # print("share reward")
        else:
            reward_n = reward_n + global_reward
            # print("not share reward")

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        if self.current_step >= self.max_step:
            done_n = [True] * self.n 

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.transporters

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        # print("env reset")
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent)

    # get global reward of the harvesters
    def _get_global_reward(self):
        glo_rew = 0
        for harv in self.world.harvesters:
            if harv.full:
                glo_rew -= 2
            if harv.transporting:
                glo_rew += 5
            if harv.complete_task:
                glo_rew += 10
            if harv.moving:
                glo_rew += 0.1
        return glo_rew

    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        agent.vel = action

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', close=False):
        if close:
            # close any existic renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        for i in range(len(self.viewers)):
            # create viewers (if necessary)

            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from . import rendering
                self.viewers[i] = rendering.Viewer(600, 800)
                self.viewers[i].set_bounds(-300, 300, -100, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            for entity in self.world.harvesters:
                geom = rendering.make_circle(7, 30)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=1)
                geom.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            for entity in self.world.transporters:
                geom = rendering.make_circle(5, 30)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=1)
                geom.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # The whole field
            field = rendering.make_polygon([(0, 0), (self.world.field.field_width, 0), (self.world.field.field_width, self.world.field.field_length),(0, self.world.field.field_length)], False)
            field.set_color(0, 0, 0)
            self.render_geoms.append(field)

            # The depot
            acircle = rendering.make_circle(10,30)
            trans = rendering.Transform(translation=self.world.field.depot)
            acircle.set_color(0, 0, 0)
            acircle.add_attr(trans)
            self.render_geoms.append(acircle)

            # The harvesting area
            harv_area = rendering.make_polygon([(0, self.world.field.headland_width), (0, self.world.field.field_length - self.world.field.headland_width), (self.world.field.field_width, self.world.field.field_length - self.world.field.headland_width), (self.world.field.field_width, self.world.field.headland_width)], False)
            # harv_area.set_color(0.941, 1, 0.941)
            harv_area.set_color(0, 1, 0)
            harv_area.set_linewidth(4)
            self.render_geoms.append(harv_area)

            # The headland
            headland1 = rendering.make_polygon([(0, 0), (0, self.world.field.headland_width), (self.world.field.field_width, self.world.field.headland_width), (self.world.field.field_width, 0)], False)
            headland2 = rendering.make_polygon([(0, self.world.field.field_length - self.world.field.headland_width), (0, self.world.field.field_length), (self.world.field.field_width, self.world.field.field_length), (self.world.field.field_width, self.world.field.field_length - self.world.field.headland_width)], False)
            headland1.set_color(0.5, 0.164, 0.164)
            headland1.set_linewidth(4)
            headland2.set_color(0.5, 0.164, 0.164)
            headland2.set_linewidth(4)
            self.render_geoms.append(headland1)
            self.render_geoms.append(headland2)

            # The ridges
            # for center in self.world.field.ridges:
            #     ridge = rendering.make_polygon_with_clw(center[0], center[1], self.world.field.ridge_width, self.world.field.ridge_length)
            #     # print(self.world.field.ridge_width)
            #     ridge.set_color(1, 0.921, 0.804)
            #     self.render_geoms.append(ridge)

            # The harvesters' trajectory
            for harv in self.world.harvesters:
                # points = harv.nav_points
                line = rendering.make_polyline(harv.nav_points)
                line.set_color(*harv.color)
                self.render_geoms.append(line)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from . import rendering

            # # The whole field
            # field = rendering.make_polygon([(0, 0), (self.world.field.field_width, 0), (self.world.field.field_width, self.world.field.field_length),(0, self.world.field.field_length)], False)
            # field.set_color(0, 0, 0)
            # self.viewers[i].add_geom(field)

            # # The depot
            # acircle = rendering.make_circle(10,30)
            # trans = rendering.Transform(translation=self.world.field.depot)
            # acircle.set_color(0, 0, 0)
            # acircle.add_attr(trans)
            # self.viewers[i].add_geom(acircle)

            # # The harvesting area
            # harv_area = rendering.make_polygon([(0, self.world.field.headland_width), (0, self.world.field.field_length - self.world.field.headland_width), (self.world.field.field_width, self.world.field.field_length - self.world.field.headland_width), (self.world.field.field_width, self.world.field.headland_width)], True)
            # harv_area.set_color(0.941, 1, 0.941)
            # self.viewers[i].add_geom(harv_area)

            # # The harvesters' trajectory
            # for harv in self.world.harvesters:
            #     # points = harv.nav_points
            #     self.viewers[i].draw_polyline(harv.nav_points, color = harv.color)

            # self.viewers[i].set_bounds(
            #     pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions

            for e, entity in enumerate(self.world.harvesters + self.world.transporters):
                self.render_geoms_xform[e].set_translation(*entity.pos)
                self.render_geoms[e].set_color(*entity.color)

            # # The cars 
            # for entity in self.world.harvesters + self.world.transporters:
            #     acircle = rendering.make_circle(5, 30)
            #     # print(entity.pos)
            #     trans = rendering.Transform(translation=entity.pos)
            #     acircle.add_attr(trans)
            #     acircle.set_color(entity.color[0], entity.color[1], entity.color[2])
            #     self.viewers[i].add_geom(acircle)
            
            # render to display or array
            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx
    
    def render_world(self, mode='human', close=False):
        from . import rendering
        metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 2
        }

        self.viewer = rendering.Viewer(600, 800)
        self.viewer.set_bounds(-300, 300, -100, 700)

        # The whole field
        field = rendering.make_polygon([(0, 0), (self.world.field.field_width, 0), (self.world.field.field_width, self.world.field.field_length),(0, self.world.field.field_length)], False)
        field.set_color(0, 0, 0)
        self.viewer.add_geom(field)

        # The depot
        acircle = rendering.make_circle(10,30)
        trans = rendering.Transform(translation=self.world.field.depot)
        acircle.set_color(0, 0, 0)
        acircle.add_attr(trans)
        self.viewer.add_geom(acircle)

        # The harvesting area
        harv_area = rendering.make_polygon([(0, self.world.field.headland_width), (0, self.world.field.field_length - self.world.field.headland_width), (self.world.field.field_width, self.world.field.field_length - self.world.field.headland_width), (self.world.field.field_width, self.world.field.headland_width)], True)
        harv_area.set_color(0, 1, 0, 0.25)
        self.viewer.add_geom(harv_area)

        # The cars 
        for entity in self.world.harvesters + self.world.transporters:
            acircle = rendering.make_circle(5, 30)
            print(entity.pos)
            trans = rendering.Transform(translation=entity.pos)
            acircle.add_attr(trans)
            acircle.set_color(entity.color[0], entity.color[1], entity.color[2])
            self.viewer.add_geom(acircle)

        # The harvesters' trajectory
        for i, harv in enumerate(self.world.harvesters):
            # points = harv.nav_points
            self.viewer.draw_polyline(harv.nav_points, color = harv.color)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
