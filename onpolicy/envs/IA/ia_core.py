import numpy as np
import math
import seaborn as sns
from numpy import random
import copy
from collections import defaultdict
from onpolicy.envs.IA.utils import compute_dist, pnpoly, a_star, heuristic, find_target_index, test_graph, find_target

# 定义农田类
class Field(object):
    def __init__(self, field_length = 310, field_width = 100, working_width = 3, yeild_per_m2 = 0.8, headland_width = 5, depot_pos = "random"):
        # farm properties
        self.field_length = field_length    # 农田长度（m）
        self.field_width = field_width      # 农田宽度（m）
        self.working_width = working_width  # 作业行宽度 (m)
        self.yeild_per_m2 = yeild_per_m2    # 单位面积产量 (kg/m2)
        self.headland_width = headland_width    # 地头宽度

        depot = np.array([[0,0], [self.field_width, 0], [0, self.field_length], [self.field_width, self.field_length]])
        depot_nav_point = np.array([[0,self.headland_width/2], [self.field_width, self.headland_width/2], \
                                    [0, self.field_length - self.headland_width/2], [self.field_width, self.field_length - self.headland_width/2]])
        if depot_pos == "random":
            randInt = random.randint(0,4)
            randIntops = (randInt + 2) % 4
            self.depot = depot[randInt, :]
            self.depot_nav_point = depot_nav_point[randInt, :]
            self.depot_nav_point_ops = depot_nav_point[randIntops, :]
        elif depot_pos == "fixed":
            self.depot = depot[0, :]
            self.depot_nav_point = depot_nav_point[0, :]
            self.depot_nav_point_ops = depot_nav_point[2, :]
        else: 
            raise NotImplementedError
        self.num_working_lines = math.ceil(self.field_width / self.working_width) # 向上取整
        self.working_line_length = self.field_length - self.headland_width 
        self.crop_x_range = [0, self.field_width]
        self.crop_y_range = [self.headland_width, self.field_length - self.headland_width]

        self.compute_nav_points()

    def compute_nav_points(self):
        # 计算农田的导航点
        # 返回: [2, num_working_lines, 2] numpy array. 
        x_coor_1 = [self.working_width/2 + i * self.working_width for i in range(self.num_working_lines - 1)]
        x_coor_1.append(self.field_width - self.working_width / 2)  #最后一个作业行用来封边
        y_coor_1 = [self.headland_width/2 for _ in range(self.num_working_lines)]
        x_coor_2 = x_coor_1[:]
        y_coor_2 = [self.field_length - self.headland_width/2 for _ in range(self.num_working_lines)]
        self.nav_points = np.array([[x_coor_1, y_coor_1],[x_coor_2, y_coor_2]]).transpose(0,2,1)

# 定义新的农田类，农田可以是不规则形状，但是需要保证两条边平行
# 需要保证左侧边和右侧边平行
class FieldIr(object):
    def __init__(self, vertices: np.array,  working_width = 3, yeild_per_m2 = 0.8, headland_width = 5, depot_pos = "random"):
        # farm properties
        assert vertices.shape == (4,2), "Wrong vertices shape!"  # 农场的四个顶点，分别是左下、左上、右上和右下顶点，需要保证左右两条边平行
        assert vertices[0][0] == vertices[1][0] and vertices[2][0] == vertices[3][0], "The edges must be parallel"
        assert vertices[0][1] < vertices[1][1] and vertices[2][1] > vertices[3][1]
        # 将农田的边平移到合适的位置
        delta_val = np.array([vertices[0][0], min(vertices[0][1], vertices[3][1])])
        self.vertices = vertices - delta_val   # 农田的四个顶点
        self.working_width = working_width  # 作业行宽度 (m)
        self.yeild_per_m2 = yeild_per_m2    # 单位面积产量 (kg/m2)
        self.headland_width = headland_width    # 地头宽度
        self.field_width = self.vertices[3][0] - self.vertices[0][0]
        self.num_working_lines = math.ceil(self.field_width / self.working_width) # 向上取整
        # depot = self.vertices.copy()
        self.vertices_nav_point = np.array([[self.vertices[0][0], self.vertices[0][1] + self.headland_width/2], \
                                       [self.vertices[1][0], self.vertices[1][1] - self.headland_width/2], \
                                       [self.vertices[2][0], self.vertices[2][1] - self.headland_width/2], \
                                       [self.vertices[3][0], self.vertices[3][1] + self.headland_width/2]])

        if depot_pos == "random":
            randInt = np.random.randint(4)
            self.randInt = randInt
            if randInt == 0 or randInt == 1:
                randIntops = (randInt + 1) % 2
            else:
                randIntops = 2 if randInt == 3 else 3
            self.depot = self.vertices[randInt]
            self.depot_nav_point = self.vertices_nav_point[randInt]
            self.depot_nav_point_ops = self.vertices_nav_point[randIntops]
            if randInt == 0 or randInt == 3:
                self.start_side = 0
            else:
                self.start_side = 1
        elif depot_pos == "fixed":
            self.randInt = 0
            self.depot = self.vertices[0]
            self.depot_nav_point = self.vertices_nav_point[0]
            self.depot_nav_point_ops = self.vertices_nav_point[1]
            self.start_side = 0
        else: 
            raise NotImplementedError
        
        self.compute_nav_points()
        self.create_dynamic_graph()

    def compute_nav_points(self):
        # 计算农田的导航点
        # 返回: [2, num_working_lines, 2] numpy array. 
        x_coor_1 = [self.vertices[0][0] + self.working_width/2 + i * self.working_width for i in range(self.num_working_lines - 1)]
        x_coor_1.append(self.vertices[3][0] - self.working_width / 2)  #最后一个作业行用来封边
        k1 = (self.vertices[3][1] - self.vertices[0][1]) / self.field_width
        y_coor_1 = [self.vertices[0][1] + self.headland_width / 2 + (x - self.vertices[0][0]) * k1 for x in x_coor_1]

        x_coor_2 = x_coor_1[:]
        k2 = (self.vertices[2][1] - self.vertices[1][1]) / self.field_width
        y_coor_2 = [self.vertices[1][1] -  self.headland_width / 2 + (x - self.vertices[1][0]) * k2 for x in x_coor_2]
        self.nav_points = np.array([[x_coor_1, y_coor_1],[x_coor_2, y_coor_2]]).transpose(0,2,1)
        # print(self.nav_points)

        
    def create_dynamic_graph(self):
        graph = defaultdict(list)
        # depot到顶点导航点
        graph[tuple(self.depot.copy())].append(tuple(self.depot_nav_point.copy()))
        graph[tuple(self.depot_nav_point.copy())].append(tuple(self.depot.copy()))
        # 顶点导航点到作业行
        graph[tuple(self.vertices_nav_point[0].copy())].append(tuple(self.nav_points[0,0,:].copy()))
        graph[tuple(self.nav_points[0,0,:].copy())].append(tuple(self.vertices_nav_point[0].copy()))
        graph[tuple(self.vertices_nav_point[1].copy())].append(tuple(self.nav_points[1,0,:].copy()))
        graph[tuple(self.nav_points[1,0,:].copy())].append(tuple(self.vertices_nav_point[1].copy()))
        graph[tuple(self.vertices_nav_point[2].copy())].append(tuple(self.nav_points[1,-1,:].copy()))
        graph[tuple(self.nav_points[1,-1,:].copy())].append(tuple(self.vertices_nav_point[2].copy()))
        graph[tuple(self.vertices_nav_point[3].copy())].append(tuple(self.nav_points[0,-1,:].copy()))
        graph[tuple(self.nav_points[0,-1,:].copy())].append(tuple(self.vertices_nav_point[3].copy()))
        # 顶点导航点之间
        graph[tuple(self.vertices_nav_point[0].copy())].append(tuple(self.vertices_nav_point[1].copy()))
        graph[tuple(self.vertices_nav_point[1].copy())].append(tuple(self.vertices_nav_point[0].copy()))
        graph[tuple(self.vertices_nav_point[2].copy())].append(tuple(self.vertices_nav_point[3].copy()))
        graph[tuple(self.vertices_nav_point[3].copy())].append(tuple(self.vertices_nav_point[2].copy()))
        for i in range(self.num_working_lines - 1):
            graph[tuple(self.nav_points[0,i,:].copy())].append(tuple(self.nav_points[0,i+1,:].copy()))
            graph[tuple(self.nav_points[0,i+1,:].copy())].append(tuple(self.nav_points[0,i,:].copy()))
            graph[tuple(self.nav_points[1,i,:].copy())].append(tuple(self.nav_points[1,i+1,:].copy()))
            graph[tuple(self.nav_points[1,i+1,:].copy())].append(tuple(self.nav_points[1,i,:].copy()))

        self.graph = graph

        # for k, v in self.graph.items():
        #     print(k, v)

# 定义收割机类
class Harvester(object):
    def __init__(self, field: FieldIr, speed = 1.5, capacity = 1800, transporting_speed = 200, dt = 0.1):
        self.id = 0
        self.name = ''
        self.color = None
        self.field = field
        self.speed = float(speed)   # 收割机的运行速度(m/s)
        self.capacity = float(capacity)    # 容量，以kg为单位，1800kg约折合2.4立方米小麦。
        self.transporting_speed = transporting_speed    # 转运速度，kg/s
        self.dt = dt
        self.yeild_per_second = self.field.yeild_per_m2 * self.field.working_width * self.speed    # 计算得到收割机以最大速度行驶时每秒的产量(kg/s)，default得到的是3.6
        
        self.pos = self.field.depot  # 所有收割机都初始化在粮仓位置
        self.time = 0.0 # 记录当前时刻
        self.last_trans_time = 0.0  # 记录上一次转运完成时间
        self.total_wait_time = 0.0  # 记录总的等待时间
        self.new_wait_time = 0.0    # 记录新增加的等待时间
        self.load = 0.0 # 当前收割机的总负载
        self.cur_working_line = -1  # 开始时的作业行标识为-1
        self.load_percent = self.load / self.capacity
        self.complete_traj = False
        self.has_a_trans = False
        self.able_to_trans = False
        self.chosen = False

    def dispatch_tasks(self, working_lines):
        self.working_lines = working_lines  # the working lines
        nav_points = []
        nav_points.append(self.field.depot)
        nav_points.append(self.field.depot_nav_point)   # 从机库出发
        if not self.field.start_side: # start side = 0
            for i in range(len(self.working_lines)):
                nav_points.append(self.field.nav_points[i % 2, self.working_lines[i], :])
                nav_points.append(self.field.nav_points[(i + 1) % 2, self.working_lines[i], :])
        else:   # start side = 1
            for i in range(len(self.working_lines)):
                nav_points.append(self.field.nav_points[(i + 1) % 2, self.working_lines[i], :])
                nav_points.append(self.field.nav_points[i % 2, self.working_lines[i], :])
        if len(self.working_lines) % 2 == 1:
            nav_points.append(self.field.depot_nav_point_ops)   # 如果奇数个作业行，农机会停在对面位置，需要先导航到机库对面的导航点再回机库
            nav_points.append(self.field.depot) # 回到机库
        else:
            nav_points.append(self.field.depot_nav_point)   # 如果偶数作业行，返回临近机库的导航点，再回到机库
            nav_points.append(self.field.depot) # 回到机库
        self.nav_points = np.array(nav_points)
        # self.nav_points = nav_points

        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)

    # def in_harvest_field(self):
    #     if self.dir[1] != 1 and self.dir[1] != -1:
    #         return False
    #     if self.cur_working_line == -1:
    #         return False
    #     if abs(self.pos[1] - self.curr_nav_point[1]) <= self.field.headland_width/2 or abs(self.pos[1] - self.old_nav_point[1]) <= self.field.headland_width/2:
    #         return False
    #     return True
    
    def in_harvest_field(self):
        d = np.array([0, self.field.headland_width / 2])
        return pnpoly(np.array([self.field.vertices_nav_point[0] + d, self.field.vertices_nav_point[1] - d, self.field.vertices_nav_point[2] - d, self.field.vertices_nav_point[3] + d]), self.pos)

    def in_head_lines(self):
        p1 = self.field.vertices[0] + np.array([0, self.field.headland_width / 2])
        p2 = self.field.vertices[3] + np.array([0, self.field.headland_width / 2])
        a1 = p1 - self.pos
        a2 = p2 - self.pos
        if np.cross(a1, a2) == 0:
            return True
        p3 = self.field.vertices[1] - np.array([0, self.field.headland_width / 2])
        p4 = self.field.vertices[2] - np.array([0, self.field.headland_width / 2])
        a3 = p3 - self.pos
        a4 = p4 - self.pos
        if np.cross(a3, a4) == 0:
            return True
        return False
    
    def move(self):
        if self.complete_traj: 
            return
        
        # 更新自身位置
        pred_new_pos = self.pos + self.dir * self.dt * self.speed
        while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
            left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
            self.nav += 1
            # 更新作业行
            if 3 <= self.nav < len(self.working_lines) * 2 + 2 and self.nav % 2 == 1:
                self.cur_working_line = self.working_lines[math.floor((self.nav - 2) / 2)]
            # 更新连接图
            elif 3 <= self.nav <= len(self.working_lines) * 2 + 2 and self.nav % 2 == 0:
                self.cur_working_line = -1
                self.field.graph[tuple(self.nav_points[self.nav - 2].copy())].append(tuple(self.nav_points[self.nav - 1].copy()))
                self.field.graph[tuple(self.nav_points[self.nav - 1].copy())].append(tuple(self.nav_points[self.nav - 2].copy()))
                # print(self.nav_points[self.nav - 1], self.nav_points[self.nav - 2])
            else:
                self.cur_working_line = -1

            if self.nav == len(self.nav_points):    #complete task
                self.pos =self.nav_points[-1]
                self.complete_traj = True
                return
            self.curr_nav_point = self.nav_points[self.nav]
            self.old_nav_point = self.nav_points[self.nav - 1]
            self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
            pred_new_pos = self.old_nav_point + left_dis * self.dir
        self.pos = pred_new_pos

        # 更新作业行信息
        # if 0 <= self.nav - 2 < len(self.working_lines) * 2:
        #     if self.cur_working_line != self.working_lines[math.floor((self.nav - 2) / 2)]:
        #         self.last_working_line = self.cur_working_line
        #         self.cur_working_line = self.working_lines[math.floor((self.nav - 2) / 2)]
        # elif self.nav - 2 >= len(self.working_lines) * 2:
        #     self.last_working_line = self.working_lines[-1]
        #     self.cur_working_line = -1

        # 更新粮仓储量
        if self.in_harvest_field():
            self.load += self.yeild_per_second * self.dt
            if self.load > self.capacity:
                self.load = self.capacity
        self.load_percent = self.load / self.capacity

    def transport(self):
        self.load = max(0.0, self.load - self.transporting_speed * self.dt)
        self.load_percent = self.load / self.capacity
        if self.load_percent == 0.0 and self.able_to_trans:
            self.able_to_trans = False
            self.last_trans_time = self.time

    def update_state(self):
        self.time += self.dt
        old_wait_time = self.total_wait_time

        if self.complete_traj:
            self.able_to_trans = False
            # 将收割机的粮仓清空，防止其它运粮车来卸粮
            self.load = 0.0
            self.load_percent = 0.0
            return

        if self.time - self.last_trans_time < 60:   # 两次转运的间隔大于60s
            self.able_to_trans = False
        else:
            self.able_to_trans = True

        if self.has_a_trans and self.able_to_trans:
            self.transport()
        elif self.load_percent != 1.0:
            self.move()
        else:
            self.total_wait_time += self.dt # 产生等待时间

        self.new_wait_time = self.total_wait_time - old_wait_time

    def get_state(self):    # 13 dim
        # state = np.concatenate([[self.id], self.pos, self.dir, [self.capacity - self.load], [self.load_percent], \
        #                         [int(self.able_to_trans)], [self.cur_working_line], [self.last_trans_time], [self.time - self.last_trans_time]])
        state = np.concatenate([[self.id], [self.speed], np.array(self.pos) / 100, self.dir, [(self.capacity - self.load) / 100],\
                                 np.array(self.curr_nav_point) / 100, np.array(self.old_nav_point) / 100, [float(self.chosen)], \
                                [float(self.complete_traj)], [(self.time - self.last_trans_time) / 100]])
        return state

class Transporter(object):
    def __init__(self, field: FieldIr, speed = 6, capacity = 8000, transporting_speed = 200, dt = 0.1, pos_error = 2):
        self.id = 0
        self.name = ''
        self.color = None
        self.field = field
        self.dt = dt
        self.transporting_speed = transporting_speed    # 转运速度，kg/s，需要和harvester的转运速度一致
        self.capacity = float(capacity) # 容量，单位kg，运粮车的容量要明显大于收割机
        self.speed = float(speed)   # 行驶速度，单位m/s
        self.pos_error = pos_error  # 认为收割机和运粮车相距多远即可开始转运。需要根据二者速度和dt计算：(v_1 + v_2) * dt / 2
        
        self.total_trip = 0.0   #总行驶路程，单位m
        self.trans_times: int = 0   #总转运次数
        self.nav_points =[]
        self.pos = self.field.depot # 运粮车初始位置在粮仓
        self.dir = np.zeros(2)
        self.load = 0.0
        self.load_percent = self.load / self.capacity
        self.has_dispatch_task = False  # 当前是否有调运任务。调运任务包括返回机库卸粮和前往指定收割机转运
        # 返回机库卸载粮食
        self.returning_to_depot = False # 当前是否在返回机库
        self.unloading = False  # 当前是否在机库卸粮
        # 前往指定收割机分成三个阶段：寻找收割机，转运，回到地头
        self.searching_for_harv = False
        self.transporting = False
        self.returning_to_headland = False
        self.last_nav_point = np.zeros(2)   # 保存上一个导航点
        self.serving_harv = None

        self.new_trip_len = 0.0
        self.new_trans_times = 0
    
    def get_state(self):    # 5 dim
        # state = np.concatenate([[self.id], self.pos, self.dir, [self.capacity - self.load], [self.load_percent], \
        #                         [float(self.has_dispatch_task)], [float(self.returning_to_depot)], [float(self.unloading)], \
        #                         [float(self.searching_for_harv)], [float(self.transporting)], [float(self.returning_to_headland)]])
        state = np.concatenate([[self.id], [self.speed], np.array(self.pos) / 100, [(self.capacity - self.load) / 100], [float(self.has_dispatch_task)]])
        return state

    def add_nav_point(self, point):
        assert point.size == 2, "The point must has 2 dimensions."
        if len(self.nav_points) > 0 and np.all(self.nav_points[-1] == point): return    # 最后一个导航点和新导航点不重合
        self.nav_points.append(point)
        # assert point[0] == self.nav_points[-1][0] or point[1] == self.nav_points[-1][1], "The harvester can only move Horizontally and Vertically."

    def vehicle_in_north_side(self):
        p1 = np.array([self.field.vertices[0][0], (self.field.vertices[0][1] + self.field.vertices[1][1]) / 2])
        p2 = np.array([self.field.vertices[2][0], (self.field.vertices[2][1] + self.field.vertices[3][1]) / 2])
        a1 = p1 - self.pos
        a2 = p2 - self.pos
        return np.cross(a1, a2) > 0

    def point_in_north_side(self, point):
        p1 = np.array([self.field.vertices[0][0], (self.field.vertices[0][1] + self.field.vertices[1][1]) / 2])
        p2 = np.array([self.field.vertices[2][0], (self.field.vertices[2][1] + self.field.vertices[3][1]) / 2])
        a1 = p1 - point
        a2 = p2 - point
        return np.cross(a1, a2) > 0

    def in_same_side(self, point):  # 判断当前车辆位置和传入的点是不是在一侧
        if self.vehicle_in_north_side() and self.point_in_north_side(point):
            return True
        elif not self.vehicle_in_north_side() and not self.point_in_north_side(point):
            return True
        else:
            return False

    def reset_nav_and_dir(self):
        assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)

    # 为returning_to_depot过程分配导航点
    def assign_return_depot_nav_points(self):
        assert self.nav_points == [], "Cannot assign nav points when former task has not finished."
        if np.all(self.pos == self.field.depot):     # 如果当前就在depot，直接return
            self.returning_to_depot = False
            return
        self.returning_to_depot = True
        self.add_nav_point(self.pos)
        if self.in_same_side(self.field.depot_nav_point):
            self.add_nav_point(self.field.depot_nav_point)
        else:
            self.add_nav_point(self.field.depot_nav_point_ops)
        self.add_nav_point(self.field.depot)
        self.reset_nav_and_dir()

    # 为searching_for_harv分配导航点
    def assign_search_nav_points(self, harv: Harvester):
        assert self.serving_harv == None, "Cannot do tasks while other tasks is doing"
        assert self.nav_points == [], "Cannot assign nav points when former task has not finished."
        if harv.chosen or harv.cur_working_line == -1:
            # 收割机已经被选择，或者刚开始工作，或者已经完成所有作业返回仓库，则不进行转运。保证收割机的 nav >= 2
            # print("Current harvester has just started or has completed task.")
            self.searching_for_harv = False
            self.serving_harv = None
            harv.chosen = False
            return
        assert harv.nav >= 2, "Strange case ???"
        self.searching_for_harv = True
        self.serving_harv = harv
        harv.chosen = True
        self.add_nav_point(self.pos)    # 每次调运任务结束都保证运粮车在地头导航点
        if np.all(self.pos == self.field.depot):
            self.add_nav_point(self.field.depot_nav_point)  # 如果出发是在机库，加入机库导航点
        harv_old_nav_point = harv.old_nav_point
        harv_curr_nav_point = harv.curr_nav_point
        # 当前位置和收割机上一个导航点在同侧，则依此加入上一个和当前导航点
        if self.in_same_side(harv_old_nav_point):  
            self.add_nav_point(harv_old_nav_point)
            self.add_nav_point(harv_curr_nav_point)
            # 再多加上两个导航点
            if harv.nav + 1 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 2])
        # old_nav_point的上一个导航点和农机在同侧，则加入上一个导航点
        elif harv.nav - 2 >= 0 and self.in_same_side(harv.nav_points[harv.nav - 2]):
            self.add_nav_point(harv.nav_points[harv.nav - 2])
            self.add_nav_point(harv_old_nav_point)
            self.add_nav_point(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 2])
            if harv.nav + 3 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 3])
        # old_nav_point的上两个个导航点和农机在同侧，则加入导航点
        elif harv.nav >= 3 and self.in_same_side(harv.nav_points[harv.nav - 3]):
            self.add_nav_point(harv.nav_points[harv.nav - 3])
            self.add_nav_point(harv.nav_points[harv.nav - 2])
            self.add_nav_point(harv_old_nav_point)
            self.add_nav_point(harv_curr_nav_point)
            if harv.nav + 1 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 1])
            if harv.nav + 2 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 2])
            if harv.nav + 3 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 3])
            if harv.nav + 4 < len(harv.nav_points):
                self.add_nav_point(harv.nav_points[harv.nav + 4])
        else:
            self.searching_for_harv = False
            self.serving_harv = None
            self.nav_points = []
            harv.chosen = False
            # print("STRANGE CASE")
            return
        
        self.reset_nav_and_dir()
        return
    
    def in_head_lines(self):
        p1 = self.field.vertices[0] + np.array([0, self.field.headland_width / 2])
        p2 = self.field.vertices[3] + np.array([0, self.field.headland_width / 2])
        a1 = p1 - self.pos
        a2 = p2 - self.pos
        if np.cross(a1, a2) == 0:
            return True
        p3 = self.field.vertices[1] - np.array([0, self.field.headland_width / 2])
        p4 = self.field.vertices[2] - np.array([0, self.field.headland_width / 2])
        a3 = p3 - self.pos
        a4 = p4 - self.pos
        if np.cross(a3, a4) == 0:
            return True
        return False
    
    def assign_return_head_nav_points(self):
        assert self.nav_points == [], "Cannot assign nav points when former task has not finished."
        if self.in_head_lines():
            self.returning_to_headland = False
            return
        self.returning_to_headland = True
        self.add_nav_point(self.pos)
        self.add_nav_point(self.last_nav_point)
        self.reset_nav_and_dir()

    def update_state(self):
        self.check_dispatching()

        old_total_trip = self.total_trip
        old_trans_time = self.trans_times
        if self.searching_for_harv:
            assert self.serving_harv != None, "The harvester must be provided."
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed # 增加总路程
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 轨迹结束都没有找到收割机，返回
                    self.pos =self.nav_points[-1]
                    print("Cannot find the harvester!!")
                    self.nav_points = []
                    self.searching_for_harv = False
                    # self.returning_to_headland = True
                    self.serving_harv.chosen = False
                    self.serving_harv = None
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos
            # 判断是否找到收割机
            if np.linalg.norm(self.pos - self.serving_harv.pos) < self.pos_error:
                self.nav_points = []
                self.searching_for_harv = False
                self.transporting = True
                self.last_nav_point = self.old_nav_point
                # self.assign_return_head_nav_points()
                if self.serving_harv.able_to_trans and self.load_percent != 1.0:
                    self.trans_times += 1

        elif self.transporting:
            assert self.serving_harv != None, "The harvester must be provided."
            if self.serving_harv.able_to_trans and self.load_percent != 1.0:
                self.serving_harv.has_a_trans = True
                self.load = min(self.capacity, self.load + self.transporting_speed * self.dt)
                self.load_percent = self.load / self.capacity
            elif self.serving_harv.able_to_trans == False:   # 收割机空了，或者本身就因为间隔太短无法卸粮食
                self.serving_harv.has_a_trans = False
                self.serving_harv.chosen = False
                self.serving_harv = None
                self.transporting = False
                self.assign_return_head_nav_points()
            else:   # self.load_percentage == 1.0，在这里可以加负的奖励
                self.serving_harv.has_a_trans = False
                self.serving_harv.chosen = False
                self.serving_harv = None
                self.transporting = False
                self.assign_return_head_nav_points()

        elif self.returning_to_headland:
            assert self.serving_harv == None, "No harvester should be provided."
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0:  # 不支持连续一个time step拐弯多次，要设置dt足够小。
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 结束导航
                    self.pos =self.nav_points[-1]
                    self.nav_points = []
                    self.returning_to_headland = False
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos

        elif self.returning_to_depot:
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed # 增加总行程
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 结束导航
                    self.pos =self.nav_points[-1]
                    self.nav_points = []
                    self.returning_to_depot = False
                    self.unloading = True
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos

        elif self.unloading:
            assert np.all(self.pos == self.field.depot), "The harvester is not in the depot!"
            self.load = max(0, self.load - self.transporting_speed * self.dt)
            self.load_percent = self.load / self.capacity
            if self.load_percent == 0:
                self.unloading = False

        self.new_trip_len = self.total_trip - old_total_trip
        self.new_trans_times = self.trans_times - old_trans_time
        # else:
        #     return
        
    def check_dispatching(self):
        if self.searching_for_harv or self.transporting or self.returning_to_headland or self.returning_to_depot or self.unloading:
            self.has_dispatch_task = True
        else:
            self.has_dispatch_task = False
    
    def set_action(self, action, harv = None):
        self.check_dispatching()
        # 如果有调运任务，直接返回
        if self.has_dispatch_task:
            return
        # action == 0, 不采取任何动作
        if action == 0: 
            return
        # action == 1， 返回depot卸载粮食
        elif action == 1:   #1，前往depot卸载粮食
            self.assign_return_depot_nav_points()
        # action >= 2，前往其它收割机卸载粮食
        elif action >= 2: 
            self.assign_search_nav_points(harv)
            # 设置转运成本

class Transporter_New(object):
    def __init__(self, field: FieldIr, speed = 6, capacity = 8000, transporting_speed = 200, dt = 0.1, pos_error = 2):
        self.id = 0
        self.name = ''
        self.color = None
        self.field = field
        self.dt = dt
        self.transporting_speed = transporting_speed    # 转运速度，kg/s，需要和harvester的转运速度一致
        self.capacity = float(capacity) # 容量，单位kg，运粮车的容量要明显大于收割机
        self.speed = float(speed)   # 行驶速度，单位m/s
        self.pos_error = pos_error  # 认为收割机和运粮车相距多远即可开始转运。需要根据二者速度和dt计算：(v_1 + v_2) * dt / 2
        
        self.total_trip = 0.0   #总行驶路程，单位m
        self.trans_times: int = 0   #总转运次数
        self.pos = self.field.depot # 运粮车初始位置在粮仓
        self.nav_points = [self.field.depot, self.field.depot_nav_point]
        self.dir = (self.field.depot_nav_point - self.field.depot) / np.linalg.norm(self.field.depot_nav_point - self.field.depot)
        self.load = 0.0
        self.load_percent = self.load / self.capacity
        self.has_dispatch_task = False  # 当前是否有调运任务。调运任务包括返回机库卸粮和前往指定收割机转运
        # 返回机库卸载粮食
        self.returning_to_depot = False # 当前是否在返回机库
        self.unloading = False  # 当前是否在机库卸粮
        # 前往指定收割机分成三个阶段：寻找收割机，转运，回到地头
        self.searching_for_harv = False
        self.transporting = False
        self.returning_to_headland = False
        self.serving_harv = None

        self.new_trip_len = 0.0
        self.new_trans_times = 0
    
    def get_state(self):    # 5 dim
        # state = np.concatenate([[self.id], self.pos, self.dir, [self.capacity - self.load], [self.load_percent], \
        #                         [float(self.has_dispatch_task)], [float(self.returning_to_depot)], [float(self.unloading)], \
        #                         [float(self.searching_for_harv)], [float(self.transporting)], [float(self.returning_to_headland)]])
        state = np.concatenate([[self.id], [self.speed], np.array(self.pos) / 100, [(self.capacity - self.load) / 100], [float(self.has_dispatch_task)]])
        return state

    def add_nav_point(self, point):
        assert point.size == 2, "The point must has 2 dimensions."
        if len(self.nav_points) > 0 and np.all(self.nav_points[-1] == point): return    # 最后一个导航点和新导航点不重合
        self.nav_points.append(point)
        # assert point[0] == self.nav_points[-1][0] or point[1] == self.nav_points[-1][1], "The harvester can only move Horizontally and Vertically."

    def vehicle_in_north_side(self):
        p1 = np.array([self.field.vertices[0][0], (self.field.vertices[0][1] + self.field.vertices[1][1]) / 2])
        p2 = np.array([self.field.vertices[2][0], (self.field.vertices[2][1] + self.field.vertices[3][1]) / 2])
        a1 = p1 - self.pos
        a2 = p2 - self.pos
        return np.cross(a1, a2) > 0

    def point_in_north_side(self, point):
        p1 = np.array([self.field.vertices[0][0], (self.field.vertices[0][1] + self.field.vertices[1][1]) / 2])
        p2 = np.array([self.field.vertices[2][0], (self.field.vertices[2][1] + self.field.vertices[3][1]) / 2])
        a1 = p1 - point
        a2 = p2 - point
        return np.cross(a1, a2) > 0

    def in_same_side(self, point):  # 判断当前车辆位置和传入的点是不是在一侧
        if self.vehicle_in_north_side() and self.point_in_north_side(point):
            return True
        elif not self.vehicle_in_north_side() and not self.point_in_north_side(point):
            return True
        else:
            return False

    def reset_nav_and_dir(self):
        assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
        self.nav = 1    # curr nav point
        self.old_nav_point = self.nav_points[self.nav - 1]
        self.curr_nav_point = self.nav_points[self.nav]
        self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)

    def in_harvest_field(self):
        return pnpoly(self.field.vertices_nav_point, self.pos)

    def search_path(self, target, g = None):
        # assert len(self.nav_points) == 2, "Before search the vehicle should have two nav points!"
        if g == None:
            g = copy.deepcopy(self.field.graph)
        # 将当前运粮车位置和运粮车旧导航点相连
        if tuple(self.nav_points[0].copy()) not in g[tuple(self.pos.copy())]:
            g[tuple(self.pos.copy())].append(tuple(self.nav_points[0].copy()))
        if tuple(self.pos.copy()) not in g[tuple(self.nav_points[0].copy())]:
            g[tuple(self.nav_points[0].copy())].append(tuple(self.pos.copy()))
        if not (find_target(self.nav_points[0].copy(), self.field.nav_points) or \
                find_target(self.nav_points[0].copy(), self.field.vertices_nav_point)):
            # print("CONNECT")
            g[tuple(self.pos.copy())].append(tuple(self.nav_points[1].copy()))
            g[tuple(self.nav_points[1].copy())].append(tuple(self.pos.copy()))
        # 如果当前所在的作业行已经被收割，则将运粮车所在位置与目标导航点相连
        if tuple(self.nav_points[1].copy()) in g[tuple(self.nav_points[0].copy())]:
            g[tuple(self.pos.copy())].append(tuple(self.nav_points[1].copy()))
            g[tuple(self.nav_points[1].copy())].append(tuple(self.pos.copy()))
        # print("SEARCH: ", self.pos, target)
        path, _ = a_star(g, tuple(self.pos.copy()), tuple(target.copy()), heuristic)
        # if path == None:
        #     print(self.field.nav_points)
        #     print(self.field.vertices_nav_point)
        #     print(self.nav_points[0])
        #     print(self.nav_points[0] in self.field.nav_points or self.nav_points[0] in self.field.vertices_nav_point)
        #     test_graph(self.field.graph, [self.pos,target])
        #     print(self.nav_points)
        #     print(self.pos, self.field.depot, target)
        assert path != None, "Cannot find path!"
        return np.array(path)

    # 为returning_to_depot过程分配导航点
    def assign_return_depot_nav_points(self):
        assert len(self.nav_points) == 2, "Cannot assign nav points when former task has not finished."
        if np.all(self.pos == self.field.depot):     # 如果当前就在depot，直接return
            self.returning_to_depot = False
            return
        self.returning_to_depot = True
        path = self.search_path(self.field.depot)
        self.nav_points = []
        for p in path:
            self.add_nav_point(np.array(p))
        self.reset_nav_and_dir()

    # 为searching_for_harv分配导航点
    def assign_search_nav_points(self, harv: Harvester):
        assert self.serving_harv == None, "Cannot do tasks while other tasks is doing"
        assert len(self.nav_points) == 2, "Cannot assign nav points when former task has not finished."
        if harv.chosen or harv.nav < 2:
            # 收割机已经被选择，或者刚开始工作，或者已经完成所有作业返回仓库，则不进行转运。保证收割机的 nav >= 2
            # print("Current harvester has just started or has completed task.")
            self.searching_for_harv = False
            self.serving_harv = None
            # harv.chosen = False
            # print("CANNOT assign")
            return
        self.searching_for_harv = True
        self.serving_harv = harv
        harv.chosen = True
        path = self.search_path(harv.old_nav_point)
        self.nav_points = []
        for p in path:
            self.add_nav_point(np.array(p))
        self.nav_points.append(harv.curr_nav_point)
        if harv.nav + 1 < len(harv.nav_points):
            self.add_nav_point(harv.nav_points[harv.nav + 1])
        if harv.nav + 2 < len(harv.nav_points):
            self.add_nav_point(harv.nav_points[harv.nav + 2])
        if harv.nav + 3 < len(harv.nav_points):
            self.add_nav_point(harv.nav_points[harv.nav + 3])
        if harv.nav + 4 < len(harv.nav_points):
            self.add_nav_point(harv.nav_points[harv.nav + 4])
        
        self.reset_nav_and_dir()
    
    def in_head_lines(self):
        p1 = self.field.vertices[0] + np.array([0, self.field.headland_width / 2])
        p2 = self.field.vertices[3] + np.array([0, self.field.headland_width / 2])
        a1 = p1 - self.pos
        a2 = p2 - self.pos
        if np.cross(a1, a2) == 0:
            return True
        p3 = self.field.vertices[1] - np.array([0, self.field.headland_width / 2])
        p4 = self.field.vertices[2] - np.array([0, self.field.headland_width / 2])
        a3 = p3 - self.pos
        a4 = p4 - self.pos
        if np.cross(a3, a4) == 0:
            return True
        return False
    
    def update_state(self):
        self.check_dispatching()

        old_total_trip = self.total_trip
        old_trans_time = self.trans_times
        if self.searching_for_harv:
            assert self.serving_harv != None, "The harvester must be provided."
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed # 增加总路程
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 轨迹结束都没有找到收割机，返回
                    self.pos =self.nav_points[-1]
                    print("Cannot find the harvester!!")
                    self.nav_points = self.nav_points[-2:]
                    self.searching_for_harv = False
                    # self.returning_to_headland = True
                    self.serving_harv.chosen = False
                    self.serving_harv = None
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos
            # 判断是否找到收割机
            if np.linalg.norm(self.pos - self.serving_harv.pos) < self.pos_error:
                self.nav_points = [self.old_nav_point, self.curr_nav_point]
                self.searching_for_harv = False
                self.transporting = True
                # self.assign_return_head_nav_points()
                if self.serving_harv.able_to_trans and self.load_percent != 1.0:
                    self.trans_times += 1

        elif self.transporting:
            assert self.serving_harv != None, "The harvester must be provided."
            if self.serving_harv.able_to_trans and self.load_percent != 1.0:
                self.serving_harv.has_a_trans = True
                self.load = min(self.capacity, self.load + self.transporting_speed * self.dt)
                self.load_percent = self.load / self.capacity
            elif self.serving_harv.able_to_trans == False:   # 收割机空了，或者本身就因为间隔太短无法卸粮食
                self.serving_harv.has_a_trans = False
                self.serving_harv.chosen = False
                self.serving_harv = None
                self.transporting = False
            else:   # self.load_percentage == 1.0，在这里可以加负的奖励
                self.serving_harv.has_a_trans = False
                self.serving_harv.chosen = False
                self.serving_harv = None
                self.transporting = False

        elif self.returning_to_depot:
            assert len(self.nav_points) >= 2, "The nav_points list must have at least two points."
            pred_new_pos = self.pos + self.dir * self.dt * self.speed
            self.total_trip += self.dt * self.speed # 增加总行程
            while (pred_new_pos - self.old_nav_point) @ (pred_new_pos - self.curr_nav_point) > 0: 
                left_dis = np.linalg.norm(pred_new_pos - self.curr_nav_point)
                self.nav += 1
                if self.nav == len(self.nav_points):    # 结束导航
                    self.pos =self.nav_points[-1]
                    self.nav_points = self.nav_points[-2:]
                    self.returning_to_depot = False
                    self.unloading = True
                    self.trans_times += 1
                    return
                self.curr_nav_point = self.nav_points[self.nav]
                self.old_nav_point = self.nav_points[self.nav - 1]
                self.dir = (self.curr_nav_point - self.old_nav_point) / np.linalg.norm(self.curr_nav_point - self.old_nav_point)
                pred_new_pos = self.old_nav_point + left_dis * self.dir
            self.pos = pred_new_pos

        elif self.unloading:
            assert np.all(self.pos == self.field.depot), "The harvester is not in the depot!"
            self.load = max(0, self.load - self.transporting_speed * self.dt)
            self.load_percent = self.load / self.capacity
            if self.load_percent == 0:
                self.unloading = False

        self.new_trip_len = self.total_trip - old_total_trip
        self.new_trans_times = self.trans_times - old_trans_time
        # else:
        #     return
        
    def check_dispatching(self):
        if self.searching_for_harv or self.transporting or self.returning_to_headland or self.returning_to_depot or self.unloading:
            self.has_dispatch_task = True
        else:
            self.has_dispatch_task = False
    
    def set_action(self, action, harv = None):
        self.check_dispatching()
        # 如果有调运任务，直接返回
        if self.has_dispatch_task:
            return
        # action == 0, 不采取任何动作
        if action == 0: 
            return
        # action == 1， 返回depot卸载粮食
        elif action == 1:   #1，前往depot卸载粮食
            self.assign_return_depot_nav_points()
        # action >= 2，前往其它收割机卸载粮食
        elif action >= 2: 
            self.assign_search_nav_points(harv)
            # 设置转运成本

            
# multi-agent world
class World(object):
    def __init__(self, args):
        # farm properties
        self.world_step = 0

        self.color_list = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0.6, 0.34],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0.9, 0], 
        [1, 0.5, 0],
        [1, 0.9, 0.8],
        [1, 0.5, 0.5],
        [0.097, 0.097, 0.437]
        ])

        self.harv_field_dict = {
            2: (120, 300), 
            3: (150, 350), 
            4: (180, 400), 
            5: (200, 420), 
            6: (250, 450), 
            7: (300, 500)
        }

        self.dt = args.dt
        self.decision_dt = args.decision_dt
        self.wait_time_factor = args.wait_time_factor
        self.distance_factor = args.distance_factor
        self.trans_times_factor = args.trans_times_factor
        self.num_harvester = args.num_harvester
        self.num_transporter = args.num_transporter

        self.episode_length = args.episode_length
        self.shared_reward = args.shared_reward
        self.trans_speed = args.trans_speed

        self.harv_vmin = args.harv_vmin
        self.harv_vmax = args.harv_vmax
        self.harv_capmin = args.harv_capmin
        self.harv_capmmax = args.harv_capmax
        self.trans_vmin = args.trans_vmin
        self.trans_vmax = args.trans_vmax
        self.trans_capmin = args.trans_capmin
        self.trans_capmax = args.trans_capmax
        self.field_width_max = self.harv_field_dict[args.num_harvester][0]
        self.field_length_max = self.harv_field_dict[args.num_harvester][1]

        self.reset()

    def reset(self):
        self.field = None
        self.harvesters = None
        self.transporters = None
        self.world_step = 0

        x0 = 0
        x1 = np.random.randint(self.field_width_max - 60, self.field_width_max)
        y0 = np.random.randint(0, 60)
        y1 = np.random.randint(self.field_length_max - 60, self.field_length_max)
        y2 = np.random.randint(self.field_length_max - 60, self.field_length_max)
        y3 = np.random.randint(0, 60)
        vertices = np.array([[x0, y0], [x0, y1], [x1, y2], [x1, y3]])
        self.field = FieldIr(vertices)

        self.harvesters = [Harvester(field=self.field, speed=np.random.uniform(self.harv_vmin, self.harv_vmax), \
                                     capacity=int(np.random.uniform(self.harv_capmin, self.harv_capmmax)) * 100, \
                                     transporting_speed=self.trans_speed, dt = self.dt) for _ in range(self.num_harvester)]
        self.transporters = [Transporter_New(field=self.field, speed=np.random.uniform(self.trans_vmin, self.trans_vmax), \
                                         capacity=int(np.random.uniform(self.trans_capmin, self.trans_capmax)) * 100, \
                                         transporting_speed=self.trans_speed, dt = self.dt) for _ in range(self.num_transporter)]
        for i, harv in enumerate(self.harvesters):
            harv.id = i
            harv.name = 'harvester %d' % i
        for j, trans in enumerate(self.transporters):
            trans.id = j + self.num_harvester
            trans.name = 'transporter %d' % j
        self.assign_agent_colors()

    def recover(self):
        self.field.create_dynamic_graph()
        for h in self.harvesters:
            h.pos = self.field.depot  # 所有收割机都初始化在粮仓位置
            h.time = 0.0 # 记录当前时刻
            h.last_trans_time = 0.0  # 记录上一次转运完成时间
            h.total_wait_time = 0.0  # 记录总的等待时间
            h.new_wait_time = 0.0    # 记录新增加的等待时间
            h.load = 0.0 # 当前收割机的总负载
            h.cur_working_line = -1  # 开始时的作业行标识为-1
            h.load_percent = h.load / h.capacity
            h.complete_traj = False
            h.has_a_trans = False
            h.able_to_trans = False
            h.chosen = False

            h.nav = 1    # curr nav point
            h.old_nav_point = h.nav_points[h.nav - 1]
            h.curr_nav_point = h.nav_points[h.nav]
            h.dir = (h.curr_nav_point - h.old_nav_point) / np.linalg.norm(h.curr_nav_point - h.old_nav_point)
        
        for tr in self.transporters:
            tr.total_trip = 0.0   #总行驶路程，单位m
            tr.trans_times = 0   #总转运次数
            tr.pos = tr.field.depot # 运粮车初始位置在粮仓
            tr.nav_points = [self.field.depot, self.field.depot_nav_point]
            tr.dir = (self.field.depot_nav_point - self.field.depot) / np.linalg.norm(self.field.depot_nav_point - self.field.depot)
            tr.load = 0.0
            tr.load_percent = tr.load / tr.capacity
            tr.has_dispatch_task = False  # 当前是否有调运任务。调运任务包括返回机库卸粮和前往指定收割机转运
            # 返回机库卸载粮食
            tr.returning_to_depot = False # 当前是否在返回机库
            tr.unloading = False  # 当前是否在机库卸粮
            # 前往指定收割机分成三个阶段：寻找收割机，转运，回到地头
            tr.searching_for_harv = False
            tr.transporting = False
            tr.returning_to_headland = False
            tr.serving_harv = None

            tr.new_trip_len = 0.0
            tr.new_trans_times = 0

            tr.nav = 1    # curr nav point
            tr.old_nav_point = tr.nav_points[tr.nav - 1]
            tr.curr_nav_point = tr.nav_points[tr.nav]
            tr.dir = (tr.curr_nav_point - tr.old_nav_point) / np.linalg.norm(tr.curr_nav_point - tr.old_nav_point)

    def assign_agent_colors(self, color_mode="random"):
        if color_mode == "fixed":
            harv_colors = [(0.25, 0.75, 0.25)] * self.num_harvester
            for color, agent in zip(harv_colors, self.harvesters):
                agent.color = color
            trans_colors = [(0.75, 0.25, 0.25)] * self.num_transporter
            for color, agent in zip(trans_colors, self.transporters):
                agent.color = color
        elif color_mode == "random":
            i = 0
            for harv in self.harvesters:
                harv.color = self.color_list[i % len(self.color_list)]
                i += 1
            for trans in self.transporters:
                trans.color = self.color_list[i % len(self.color_list)]
                i += 1
        else:
            raise NotImplementedError
        
    # def create_dynamic_graph(self):
    #     graph = defaultdict(list)
    #     graph['depot'].append(Edge('vertices_nav_{}'.format(self.field.randInt), self.field.headland_width / 2))
    #     for i in range(4):
    #         graph['vertices_nav_{}'.format(i)].append(Edge('vertices_nav_{}'.format((i - 1) % 4), \
    #                 compute_dist(self.field.vertices_nav_point[i], self.field.vertices_nav_point[(i - 1) % 4])))
    #         graph['vertices_nav_{}'.format(i)].append(Edge('vertices_nav_{}'.format((i + 1) % 4), \
    #                 compute_dist(self.field.vertices_nav_point[i], self.field.vertices_nav_point[(i + 1) % 4])))
    #     self.graph = graph
    #     for k, v in world.graph.items():
    #     for e in v:
    #         print(k, e.to, e.val)


    # update state of the world
    def step(self):
        self.world_step += 1
        for harv in self.harvesters:
            harv.update_state()
        for trans in self.transporters:
            trans.update_state()
                
def test_harv():
    # field = Field(field_length=500, field_width=50, working_width=3.2, depot_pos="random")
    field = Field(depot_pos="random")
    print(field.depot, field.depot_nav_point, field.depot_nav_point_ops)
    print(field.nav_points)
    har1 = Harvester(field)
    har1.reset_harv(working_lines=[2,3,4])
    print(har1.nav_points)
    for i in range(5600):
        if i % 10 == 0: # 每隔一秒打印一次位置
            state = har1.get_state()
            print(i, state)
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = True
    for i in range(400):
        if i % 10 == 0: 
            print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    # har1.has_a_trans = False
    # for i in range(5000):
    #     if i % 10 == 0:
    #         print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
    #     har1.update_state()

def test_trans():
    # field = Field(field_length=500, field_width=50, working_width=3.2, depot_pos="random")
    field = Field(depot_pos="random")
    # print(field.depot, field.depot_nav_point, field.depot_nav_point_ops)
    # print(field.nav_points)
    har1 = Harvester(field)
    har1.reset_harv(working_lines=[2,3,4])
    print(har1.nav_points)
    for i in range(5600):
        # if i % 10 == 0: # 每隔一秒打印一次位置
            # print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = True
    for i in range(400):
        # if i % 10 == 0: 
            # print(i, har1.pos, har1.load, har1.time, har1.last_trans_time, har1.able_to_trans, har1.cur_working_line)
        har1.update_state()
    har1.has_a_trans = False

    print(har1.nav, har1.old_nav_point, har1.curr_nav_point, har1.pos)
    trans1 = Transporter(field)
    trans1.assign_search_nav_points(har1)
    print(trans1.nav_points)
    print(trans1.get_state())

def test_field_ir():
    vertices = np.array([[-5,-5], [-5,50], [31, 110.5], [31, -20]])
    field = FieldIr(vertices)
    print(field.vertices)
    print(field.nav_points)
    # print(field.vertices.reshape(-1))
    p = field.nav_points[np.random.randint(2), np.random.randint(field.num_working_lines), :]
    print(p)
    print(find_target_index(field.nav_points, p))

def test_world():
    import argparse
    import numpy as np
    import time
    from onpolicy.config import get_config

    # np.random.seed(7)
    parser = get_config()
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    all_args = parser.parse_known_args()[0]
    world = World(all_args)
    print("WORLD:")
    print(world.field.depot)
    print(world.field.vertices)



if __name__ == "__main__":
    # test_harv()
    # test_trans()
    np.random.seed(0)
    test_field_ir()
    # np.random.seed(0)
    # test_world()
    # print(compute_dist(np.array([1,1]), np.array([5,4])))
