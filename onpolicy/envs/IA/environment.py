import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from onpolicy.envs.IA.ia_core import World, Harvester, Transporter, Field
import pyglet
# import os
# os.environ['DISPLAY'] = ':1'

# update bounds to center around agent
cam_range = 2

class IAMultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world: World, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True):

        self.world = world
        self.wait_time_factor = self.world.wait_time_factor
        self.distance_factor = self.world.distance_factor
        self.trans_times_factor = self.world.trans_times_factor
        self.world_length = self.world.episode_length
        self.current_step = 0
        # set required vectorized gym env property
        self.n = self.world.num_transporter
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.decision_dt = self.world.decision_dt

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
        for agent in self.world.transporters:
            self.action_space.append(spaces.Discrete(2 + self.world.num_harvester))

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
    def step(self, action_n, img_list = None, no_trans_mode = False, auto_trans_mode = False, decPt = None):
        # set action for each agent
        # action_n:a list, contains num_agent elements,each element is a (single_action_dim,)shape array. 
        self.current_step += 1

        if no_trans_mode:
            for i in range(self.world.num_harvester):
                if self.world.harvesters[i].load_percent == 1.0:
                    print(f"harv{i} full at ", self.current_step * self.world.dt, " s. ")
                    print(f"harv{i} pos: ", self.world.harvesters[i].pos)
                    print("*"*10)
                    self.world.harvesters[i].load = 0.0
                    self.world.harvesters[i].load_percent = 0.0

        elif auto_trans_mode:
            assert decPt != None, "Decision Point must be provided!"
            for i in range(self.world.num_transporter):
                if self.world.transporters[i].load_percent == 1.0:
                    self.world.transporters[i].set_action(1)
            for i in range(self.world.num_harvester):
                if self.world.harvesters[i].load_percent >= decPt:
                    if not self.world.harvesters[i].chosen:
                        pos_harv = self.world.harvesters[i].pos
                        dis_list = []
                        for j in range(self.world.num_transporter):
                            dis_list.append(np.linalg.norm(pos_harv - self.world.transporters[j].pos))
                        # print(dis_list)
                        idx = np.argsort(dis_list)
                        # print(idx)
                        for k in idx:
                            if not self.world.transporters[k].has_dispatch_task:
                                self.world.transporters[k].set_action(2, self.world.harvesters[i])
                                break

        else:
            for i, agent in enumerate(self.world.transporters):
                self._set_action(action_n[i], agent)
        # advance world state
        reward_n = [0.0 for _ in range(self.world.num_transporter)]
        reward_global = 0.0
        for _ in range(int(self.decision_dt / self.world.dt)):
            self.world.step()  # core.step()
            for i, agent in enumerate(self.world.transporters):
                new_trip_len, new_trans_times = self._get_reward(agent)
                reward_n[i] += (self.distance_factor * new_trip_len + self.trans_times_factor * new_trans_times)
            for harv in self.world.harvesters:
                reward_global -= self.wait_time_factor * harv.new_wait_time

        obs_n = []
        done_n = []
        info_n = []

        for i, agent in enumerate(self.world.transporters):
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info = {}
            # info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)
        

            # global_reward = self._get_global_reward()

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        if self.shared_reward:
            reward = np.sum(reward_n) + reward_global
            reward_n = [[reward]] * self.n
            # print("share reward")
        else:
            # print("reward_n old: ", reward_n)
            for i in range(len(reward_n)):
                reward_n[i] += reward_global
            # print("reward_n new: ", reward_n)
            # print("not share reward")

        all_complete = True
        for i in range(self.world.num_harvester):
            all_complete = all_complete and self.world.harvesters[i].complete_traj
        if all_complete:
            done_n = [True] * self.n 
            # print("All complete task. ")
            # print("Total tims: ", self.current_step * self.world.dt)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []

        for agent in self.world.transporters:
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
        return self.reward_callback(agent, self.world)

    # get global reward of the harvesters
    def _get_global_reward(self):
        glo_rew = 0
        # for harv in self.world.harvesters:
        #     if harv.full:
        #         glo_rew -= 10
        #     if harv.transporting:
        #         glo_rew += 3
        #     if harv.complete_task:
        #         glo_rew += 100
        #     if harv.moving:
        #         glo_rew += 2
        return glo_rew

    # set env action for a particular agent
    def _set_action(self, action, agent: Transporter, time=None):
        # print("Local action: ", type(action))
        act = action[0] if isinstance(action, np.ndarray) else action
        # print("action: ", action)
        if action < 2:
            agent.set_action(act)
        else:
            harvID = act - 2
            agent.set_action(2, self.world.harvesters[harvID])
        return

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', close=False, scale: int = 2):

        class DrawText:
            def __init__(self, label:pyglet.text.Label):
                self.label=label
            def render(self):
                self.label.draw()

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
                # self.viewers[i] = rendering.Viewer(600, 800)
                # self.viewers[i].set_bounds(-300, 300, -100, 700)
                self.viewers[i] = rendering.Viewer(250 * scale, 350 * scale)
                self.viewers[i].set_bounds(-10 * scale, 240 * scale , -10 * scale, 340 * scale)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            for entity in self.world.harvesters:
                geom = rendering.make_circle(2 * scale, 60)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=1)
                geom.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            for entity in self.world.transporters:
                geom = rendering.make_circle(4 * scale, 30)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=1)
                geom.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # The harvesters' completed trajectory
            for harv in self.world.harvesters:
                # points = harv.nav_points
                line = rendering.make_polyline([[0,0]])
                line.set_color(*harv.color, alpha=1)
                line.set_linewidth(1 * scale)
                self.render_geoms.append(line)

            # The whole field
            field = rendering.make_polygon([(0, 0), (self.world.field.field_width * scale, 0), \
                                            (self.world.field.field_width * scale, self.world.field.field_length * scale),\
                                            (0, self.world.field.field_length * scale)], False)
            field.set_color(0, 0, 0)
            self.render_geoms.append(field)

            # The depot
            acircle = rendering.make_circle(5 * scale, 60)
            trans = rendering.Transform(translation=np.array(self.world.field.depot) * scale)
            acircle.set_color(0, 0, 0)
            acircle.add_attr(trans)
            self.render_geoms.append(acircle)

            # The harvesting area
            harv_area = rendering.make_polygon([(0, self.world.field.headland_width * scale), \
                                                (0, self.world.field.field_length * scale - self.world.field.headland_width * scale), \
                                                (self.world.field.field_width * scale, self.world.field.field_length * scale - self.world.field.headland_width * scale), \
                                                (self.world.field.field_width * scale, self.world.field.headland_width * scale)], False)
            # harv_area.set_color(0.941, 1, 0.941)
            harv_area.set_color(0, 1, 0)
            harv_area.set_linewidth(2 * scale)
            self.render_geoms.append(harv_area)

            # The headland
            headland1 = rendering.make_polygon([(0, 0), (0, self.world.field.headland_width * scale), \
                                                (self.world.field.field_width * scale, self.world.field.headland_width * scale), \
                                                (self.world.field.field_width* scale, 0)], False)
            headland2 = rendering.make_polygon([(0, self.world.field.field_length * scale - self.world.field.headland_width * scale), \
                                                (0, self.world.field.field_length* scale), \
                                                (self.world.field.field_width * scale, self.world.field.field_length * scale), \
                                                (self.world.field.field_width * scale, self.world.field.field_length * scale - self.world.field.headland_width * scale)], False)
            headland1.set_color(0.5, 0.164, 0.164)
            headland1.set_linewidth(1 * scale)
            headland2.set_color(0.5, 0.164, 0.164)
            headland2.set_linewidth(1 * scale)
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
                nav_points = np.array(harv.nav_points.copy())
                nav_points = nav_points * scale
                line = rendering.make_polyline(nav_points)
                line.set_color(*harv.color, alpha=0.2)
                line.set_linewidth(1 * scale)
                self.render_geoms.append(line)

            # The text
            self.text_geoms = []
            for h, harv in enumerate(self.world.harvesters):
                label = pyglet.text.Label(f"Harvester {h}", font_size=10,
                                x=120*scale, y=(300-h*20/scale)*scale , anchor_x='left', anchor_y='bottom',
                                color=(0, 0, 0, 255))
                label.draw()
                self.text_geoms.append(DrawText(label))
                acircle = rendering.make_circle(2 * scale, 30)
                transl = rendering.Transform(translation=np.array([110*scale, (305-h*20/scale)*scale]))
                acircle.set_color(*harv.color)
                acircle.add_attr(transl)
                self.render_geoms.append(acircle)
            for t, trans in enumerate(self.world.transporters):
                label = pyglet.text.Label(f"Transporter {t}", font_size=10,
                                x=120*scale, y=(300-(h+1)*20/scale-t*20/scale)*scale , anchor_x='left', anchor_y='bottom',
                                color=(0, 0, 0, 255))
                label.draw()
                self.text_geoms.append(DrawText(label))
                acircle = rendering.make_circle(4 * scale, 30)
                transl = rendering.Transform(translation=np.array([110*scale, (305-(h+1)*20/scale-t*20/scale)*scale]))
                acircle.set_color(*trans.color)
                acircle.add_attr(transl)
                self.render_geoms.append(acircle)
            # add time 
            # print(self.current_step)
            label = pyglet.text.Label("Time: ", font_size=10,
                            x=120*scale, y=(300-(h+1)*20/scale-(t+1)*20/scale)*scale , anchor_x='left', anchor_y='bottom',
                            color=(0, 0, 0, 255))
            label.draw()
            self.text_geoms.append(DrawText(label))
            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for geom in self.text_geoms:
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
                self.render_geoms_xform[e].set_translation(*entity.pos * scale)
                # self.render_geoms[e].set_color(*entity.color)
            for h, harv in enumerate(self.world.harvesters):
                message = f'Harvester {h}: ' + '%.2f'%harv.load + '/' + str(harv.capacity)
                self.text_geoms[h].label.text = message
            for t, trans in enumerate(self.world.transporters):
                message = f'Transporter {t} :' + '%.2f'%trans.load + '/' + str(trans.capacity)
                self.text_geoms[h + 1 + t].label.text = message
            message = f'Time :' + '%.1f'%(self.current_step * self.decision_dt) + 's'
            self.text_geoms[h + t + 2].label.text = message
            for h, harv in enumerate(self.world.harvesters):
                self.render_geoms[e + h + 1].v = np.concatenate([harv.nav_points[:harv.nav], [harv.pos]], axis=0)*scale

            for harv in self.world.harvesters:
                # points = harv.nav_points
                nav_points = np.array(harv.nav_points.copy())
                nav_points = nav_points * scale
                line = rendering.make_polyline([nav_points[:harv.nav], harv.pos*scale])
                line.set_color(*harv.color)
                line.set_linewidth(1 * scale)
                self.render_geoms.append(line)
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
