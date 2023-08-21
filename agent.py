import carla
import time
import numpy as np
import pickle
import math
import copy
import random
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from queue import PriorityQueue


            
original_wps = pickle.load(open('shanghai_intl_circuit', 'rb'))
all_waypoints = original_wps[1:]
obs_insert_flag = 0

class Node:
    """
    Node class for dijkstra search
    """
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

class Agent():

    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.min_radius = self.find_min_radius()
        self.mode = 1
        self.paths = []
        self.new_waypoints = []
        self.counter = 0
        self.dist = 0
        self.path_planning_flag = 0
        self.sample_size = 300
        #print("minimum radius is: ", self.min_radius)

    def find_min_radius(self):
        original_wps = pickle.load(open('shanghai_intl_circuit', 'rb'))
        all_waypoints = original_wps[1:]
        temp = []
        for i in range(len(all_waypoints)-3):
            temp.append(self.findCircle(all_waypoints[i:i+3]))
        temp = np.array(temp)
        min_radius = np.min(temp[np.nonzero(temp)])
        return min_radius

    # https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
    def findCircle(self, waypoints):
        # x1, y1, z1 = waypoints[0]
        # x2, y2, z2 = waypoints[1]
        # x3, y3, z3 = waypoints[2]
        
        if len(waypoints) == 0:
            r = 0
            return r
            
        if len(waypoints) == 1:
            x1, x2, x3 = waypoints[0][0], waypoints[0][0], waypoints[0][0]
            y1, y2, y3 = waypoints[0][1], waypoints[0][1], waypoints[0][1]
            
        if len(waypoints) == 2:
            x1, y1 = waypoints[0][:2]
            x2, y2 = waypoints[1][:2]
            x3 = (waypoints[0][0] + waypoints[1][0]) / 2
            y3 = (waypoints[0][1] + waypoints[1][1]) / 2
        #print("length of waypoints is: ", len(waypoints))
        
        if len(waypoints) == 3:
            x1, y1 = waypoints[0][:2]
            x2, y2 = waypoints[1][:2]
            x3, y3 = waypoints[2][:2]
        

        x12 = x1 - x2
        x13 = x1 - x3
    
        y12 = y1 - y2
        y13 = y1 - y3
    
        y31 = y3 - y1
        y21 = y2 - y1
    
        x31 = x3 - x1
        x21 = x2 - x1
    
        # x1^2 - x3^2
        sx13 = pow(x1, 2) - pow(x3, 2)
    
        # y1^2 - y3^2
        sy13 = pow(y1, 2) - pow(y3, 2)
    
        sx21 = pow(x2, 2) - pow(x1, 2)
        sy21 = pow(y2, 2) - pow(y1, 2)

        if ((y31 == 0 and y21 == 0) or (x31 == 0 and x21 == 0) or ((y31 * x12) == (y21 * x13))):
            r = 0
            # r = 99999
            return r
    
        #print((2 * ((y31) * (x12) - (y21) * (x13))))
        #print(y31,x12,y21,x13)

        f = (((sx13) * (x12) + (sy13) *
            (x12) + (sx21) * (x13) +
            (sy21) * (x13)) // (2 *
            ((y31) * (x12) - (y21) * (x13))))
        #print(f)
                
        g = (((sx13) * (y12) + (sy13) * (y12) +
            (sx21) * (y13) + (sy21) * (y13)) //
            (2 * ((x31) * (y12) - (x21) * (y13))))
    
        c = (-pow(x1, 2) - pow(y1, 2) -
            2 * g * x1 - 2 * f * y1)

        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
        # where centre is (h = -g, k = -f) and
        # radius r as r^2 = h^2 + k^2 - c
        h = -g
        k = -f
        sqr_of_r = h * h + k * k - c

        # r is the radius
        r = round(np.sqrt(sqr_of_r), 5)

        #print("Centre = (", h, ", ", k, ")")
        #print("Radius = ", r)

        return r     

    def prm(self, curr_x, curr_y, target_x, target_y, robot_radius, transform, obs_x, obs_y):
            # robot_radius = 2.35
            rng = None
            path = []
            
            
            obs_KDTree = KDTree(np.vstack((obs_x, obs_y)).T) # vertical stack then transpose
            
            #print("1")
            if (len(obs_x) == 0 and len(obs_y) == 0):
                return path

            sample_x, sample_y = self.sample_points(curr_x, curr_y, target_x, target_y, robot_radius, obs_x, obs_y, obs_KDTree, rng, transform)

            road_map = self.generate_road_map(sample_x, sample_y,robot_radius, obs_KDTree, transform)

            path = self.dijkstra(curr_x, curr_y, target_x, target_y, road_map, sample_x, sample_y)

            return path

    def sample_points(self, curr_x, curr_y, target_x, target_y, robot_radius, obs_x, obs_y, obs_KDTree, rng, transform):

        max_x = max(obs_x)
        max_x_y = obs_y[obs_x.index(max(obs_x))]

        max_y = max(obs_y)
        max_y_x = obs_x[obs_y.index(max(obs_y))]

        min_x = min(obs_x)
        min_x_y = obs_y[obs_x.index(min(obs_x))]

        min_y = min(obs_y)
        min_y_x = obs_x[obs_y.index(min(obs_y))]
        '''
        limit_start = [(min_x+min_y_x)/2, (min_x_y+min_y)/2]
        limit_end = [(max_x+max_y_x)/2, (max_x_y+max_y)/2]

        m = (limit_end[1]-limit_start[1])/(limit_end[0]-limit_start[0])
        b = limit_start[1]-m*limit_start[0]
        X = np.linspace(min_x, max_x, 7)
        print("X================", X.tolist())
        y_above = [m*x+b + abs(random.gauss(20,5)) for x in X]
        y_below = [m*x+b - abs(random.gauss(20,5)) for x in X]

        sample_x, sample_y = [], []
        total_x = []
        total_y = []
        total_x.append(X.tolist())
        total_x.append(X.tolist())
        total_y.append(y_above)
        total_y.append(y_below)
        '''
        x_list = [max_x, max_y_x, min_x, min_y_x]
        y_list = [max_x_y, max_y, min_x_y, min_y]
        array = np.array([[x_list[i], y_list[i]] for i in range(4)])
        #print("shape of array is", array.shape)
        if max_x_y - min_y == 0 or max_x - min_y_x == 0:
            theta = 0
        else: 
            theta = np.arctan2((max_x_y - min_y), (max_x - min_y_x))
        
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta), np.cos(theta)]])
        rot_mat_inv = np.linalg.inv(rot_mat)
        sample_size = self.sample_size

        rect_vertex = np.zeros((4, 2))
        for i in range(4):
            rect_vertex[i] = rot_mat @ array[i]
        x = np.random.uniform(np.min(rect_vertex[:, 0]), np.max(rect_vertex[:, 0]), size=sample_size)
        y = np.random.uniform(np.min(rect_vertex[:, 1]), np.max(rect_vertex[:, 1]), size=sample_size)

        sample = np.zeros((sample_size, 2))
        for i in range(sample_size):
            temp = np.array([x[i], y[i]])
            sample[i] = rot_mat_inv @ temp

        total_x = sample[:, 0]
        total_y = sample[:, 1]
        '''
        if rng is None:
            rng = np.random.default_rng()
        
        while len(sample_x) <= 10:
            temp_x = (rng.random() * (max_x - min_x)) + min_x
            temp_y = (rng.random() * (max_y - min_y)) + min_y
        '''
        sample_x = []
        sample_y = []

        for i in range(sample_size):
            temp_x = total_x[i]
            temp_y = total_y[i]
            dist, index = obs_KDTree.query([temp_x, temp_y]) #find closest dist and index to temp

            if dist >= robot_radius:
                sample_x.append(temp_x)
                sample_y.append(temp_y)

        sample_x.append(curr_x)
        sample_y.append(curr_y)
        sample_x.append(target_x)
        sample_y.append(target_y)
        #print("length of samplex is ", len(sample_x))
        #print("length of sampley is ", len(sample_y))

        plt.figure()
        plt.scatter(sample_x, sample_y, color='yellow')
        plt.scatter(curr_x, curr_y, color='red')
        plt.scatter(target_x, target_y, color='blue')
        plt.scatter(obs_x, obs_y, color='black')
        plt.savefig('samples')

        return sample_x, sample_y

    def plot_road_map(self, road_map, sample_x, sample_y):  
        for i, _ in enumerate(road_map):
            for ii in range(len(road_map[i])):
                ind = road_map[i][ii]

                plt.plot([sample_x[i], sample_x[ind]],
                        [sample_y[i], sample_y[ind]], "-k")
        plt.savefig('ha')

    def generate_road_map(self, sample_x, sample_y, robot_radius, obstacle_kd_tree, transform):
        road_map = []
        n_sample = len(sample_x)
        sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

        for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
            dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
            edge_id = []

            for ii in range(1, len(indexes)):
                nx = sample_x[indexes[ii]]
                ny = sample_y[indexes[ii]]

                if not self.is_collision(ix, iy, nx, ny, robot_radius, obstacle_kd_tree, transform):
                    edge_id.append(indexes[ii])

                if len(edge_id) >= 10:
                    break

            road_map.append(edge_id)

            #self.plot_road_map(road_map, sample_x, sample_y)

        return road_map


    def is_collision(self, curr_x, curr_y, target_x, target_y, robot_radius, obstacle_kd_tree, transform):
        x = curr_x
        y = curr_y
        dx = target_x - curr_x
        dy = target_y - curr_y
        # yaw = math.atan2(target_y - curr_y, target_x - curr_x)
        yaw = transform.rotation.yaw
        d = np.sqrt(dx ** 2 + dy ** 2)

        #if d >= 30:
        #    return True

        D = robot_radius
        n_step = int(d / D)

        for i in range(n_step):
            dist, _ = obstacle_kd_tree.query([x, y])
            if dist <= robot_radius:
                return True  # collision
            x += D * math.cos(yaw)
            y += D * math.sin(yaw)

        # goal point check
        dist, _ = obstacle_kd_tree.query([target_x, target_y])
        if dist <= robot_radius:
            return True  # collision

        return False  # OK

    def dijkstra(self, curr_x, curr_y, target_x, target_y, road_map, sample_x, sample_y):

        # Node(x, y, cost, parent_index)
        start_node = Node(curr_x, curr_y, 0.0, -1)
        goal_node = Node(target_x, target_y, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[len(road_map) - 2] = start_node

        path_found = True

        while True:
            if not open_set:
                print("Cannot find path")
                path_found = False
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if c_id == (len(road_map) - 1):
                print("goal is found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]
            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for i in range(len(road_map[c_id])):
                n_id = road_map[c_id][i]
                dx = sample_x[n_id] - current.x
                dy = sample_y[n_id] - current.y
                d = math.hypot(dx, dy)
                node = Node(sample_x[n_id], sample_y[n_id],
                            current.cost + d, c_id)

                if n_id in closed_set:
                    continue
                # Otherwise if it is already in the open set
                if n_id in open_set:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id].cost = node.cost
                        open_set[n_id].parent_index = c_id
                else:
                    open_set[n_id] = node

        if path_found is False:
            return []

        # generate final course
        path = [[goal_node.x, goal_node.y, 0]]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            path.append([n.x, n.y, 0])
            parent_index = n.parent_index

        return path

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s

        # from Carla API
        '''
        https://carla.readthedocs.io/en/0.9.13/python_api/#carlavehiclecontrol
        throttle (float)
        A scalar value to control the vehicle throttle [0.0, 1.0]. Default is 0.0.
        steer (float)
        A scalar value to control the vehicle steering [-1.0, 1.0]. Default is 0.0.
        brake (float)
        A scalar value to control the vehicle brake [0.0, 1.0]. Default is 0.0.
        hand_brake (bool)
        Determines whether hand brake will be used. Default is False.
        reverse (bool)
        Determines whether the vehicle will move backwards. Default is False.
        manual_gear_shift (bool)
        Determines whether the vehicle will be controlled by changing gears manually. Default is False.
        gear (int)
        States which gear is the vehicle running on. 
        '''
        
        def change_unit(x):
            return x / 180 * np.pi

        
        # get current 
        curr_x, curr_y, curr_z = transform.location.x, transform.location.y, transform.location.z
        #print("curr_x1 = ", curr_x)
        curr_orientation = change_unit(transform.rotation.yaw)
        #print("curr_o = ", curr_orientation)
        curr_vel = np.sqrt(vel.x ** 2 + vel.y ** 2)

        
        if self.mode == 1:
            print("start mode 111111111111111111111111111111111111111111111111")
            if len(filtered_obstacles) != 0:
            # if filtered_obstacles != []:
                # get target
                target_x, target_y, target_z = waypoints[0]
                #print("target_x1 = ", target_x)

                # get obs
                obs_x = []
                obs_y = []
                for i in range(len(filtered_obstacles)):
                    obs_x.append(filtered_obstacles[i].get_location().x)
                    #print("obs_x = ", obs_x[i])
                    obs_y.append(filtered_obstacles[i].get_location().y)
                
                obs_distance = np.inf
                #print(len(obs_x))
                for i in range(len(obs_x)):
                    #print('iteration', i, "========================================")
                    temp_d = math.sqrt((curr_x - obs_x[i])**2 + (curr_y - obs_y[i])**2)  
                    if temp_d < obs_distance:
                        obs_distance = temp_d
                print("closest obstacle distance = ", obs_distance, "+++++++++++++++++++++")        
                
                if obs_distance <= 15: # time to do avoiding ///15: the distance from the closest obstacle to start avoiding program
                    print('1111111111111')  
                    
                    for i in range(len(boundary[0])):
                        obs_x.append(boundary[0][i].transform.location.x) # might be negative
                        #print("left bound x = ", -boundary[0][i].transform.location.x)
                        obs_y.append(boundary[0][i].transform.location.y)

                    for j in range(len(boundary[1])):
                        obs_x.append(boundary[1][j].transform.location.x) # might be negative
                        #print("right bound x = ", -boundary[1][j].transform.location.x)
                        obs_y.append(boundary[1][j].transform.location.y)
                    plt.figure()
                    #plt.scatter(curr_x, curr_y, color='yellow')
                    
                    
                    self.paths = self.prm(curr_x=curr_x, curr_y=curr_y, target_x=target_x, target_y=target_y, robot_radius=2, transform=transform, obs_x=obs_x, obs_y=obs_y)
                    # self.paths = self.prm(curr_x=target_x, curr_y=target_y, target_x=curr_x, target_y=curr_y, robot_radius=2, transform=transform, obs_x=obs_x, obs_y=obs_y)
                    
                    if (len(self.paths) == 0):
                        for i in range(15):
                            curr_x, curr_y, curr_z = transform.location.x, transform.location.y, transform.location.z
                            #print("curr_x2 = ", curr_x)
                            self.paths = self.prm(curr_x=curr_x, curr_y=curr_y, target_x=target_x, target_y=target_y, robot_radius=2, transform=transform, obs_x=obs_x, obs_y=obs_y)
                            # self.paths = self.prm(curr_x=target_x, curr_y=target_y, target_x=curr_x, target_y=curr_y, robot_radius=2, transform=transform, obs_x=obs_x, obs_y=obs_y)
                            if len(self.paths) != 0:
                                break
                    
                    #self.path_planning_flag = 1
                    # self.paths=[[180,30,0],[150,35,0]] 

                    if len(self.paths) != 0:
                        print("==========================================================================")
                        start_x, start_y, start_z = self.paths[-1]
                        print(start_x == curr_x)
                        plot_x, plot_y = [], []
                        for path in self.paths:
                            plot_x.append(path[0])
                            plot_y.append(path[1])
                        
                        plt.scatter(start_x, start_y, color='red')
                        plt.plot(plot_x[:], plot_y[:], color='magenta')
                        plt.scatter(plot_x[1:-1], plot_y[1:-1], color='magenta') #edges
                        # plt.scatter(plot_x[0], plot_y[0], color='green')
                        plt.scatter(plot_x[0], plot_y[0], color='blue') # goal point
                        #print("difference is", plot_x[0]-target_x)
                        plt.scatter(obs_x, obs_y, color='black')
                        #plt.scatter(target_x, target_y, color='green')
                        plt.savefig('hahaha')
                    
                    self.mode = 0
                else:
                    self.new_waypoints = waypoints
            else:
                self.new_waypoints = waypoints

        if self.mode == 0:
            #print("start mode 000000000000000000000000000000000000000000000000000000")
            path_len = len(self.paths)
            self.new_waypoints = copy.deepcopy(waypoints)
            self.new_waypoints[0] = self.paths[path_len-1-self.counter]
            #print("using waypoints", self.new_waypoints, "``````````````````````")
            curr_x, curr_y, curr_z = transform.location.x, transform.location.y, transform.location.z
            #print("curr_x3 = ", curr_x)
            target_x, target_y, target_z = self.new_waypoints[0]
            #print("target_x2 = ", target_x)
            self.dist = math.sqrt((curr_x - target_x) ** 2 + (curr_y - target_y) ** 2)

            #print("self_dist = ", self.dist, "+++++++++++++++++++++++++++++++++++++++++++")

            if self.dist < 5:
                self.counter += 1

            #print("path len = ", path_len, "--------------------- counter = ", self.counter)

            if self.counter >= (path_len-1):
                #print("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
                self.mode = 1
                self.counter = 0

        
        target_x, target_y, target_z = self.new_waypoints[0]
        print('length of new_waypoints is, ', len(self.new_waypoints))
        


        # get current 
        curr_x, curr_y, curr_z = transform.location.x, transform.location.y, transform.location.z
        curr_orientation = change_unit(transform.rotation.yaw)
        curr_vel = np.sqrt(vel.x ** 2 + vel.y ** 2)

        # get target
        # target_x, target_y, target_z = waypoints[0]

        #print("wp", waypoints[0])


        target_orientation = np.arctan2(target_y - curr_y, target_x - curr_x) #theta_ref
        target_vel = 45
        target_vx = 0.01 * np.cos(target_orientation) * target_vel
        target_vy = 0.01 * np.sin(target_orientation) * target_vel
        target_x += target_vx
        target_y += target_vy

        
        '''
        if delta_orientation < 0.001:
            delta_orientation = 0 
        elif delta_orientation > np.pi:
            delta_orientation = delta_orientation % (2 * np.pi)  
        else:
            delta_orientation = -(delta_orientation % (2 * np.pi))
        '''


        # delta = target - current
        delta_x = np.cos(target_orientation) * (target_x - curr_x) + np.sin(target_orientation) * (target_y - curr_y)
        delta_y = -np.sin(target_orientation) * (target_x - curr_x) + np.cos(target_orientation) * (target_y - curr_y)
        delta_v = target_vel - curr_vel
        delta_orientation = target_orientation - curr_orientation  #delta_theta
        # print("delta x is: ", delta_x)
        # print("delta y is: ", delta_y)
        # print("delta orientation is: ", delta_orientation)

        # PD control
        k_x, k_y, k_v, k_theta = 0.5, 1.5, 1, 1
        K = np.array([[k_x, 0, 0, k_v],
                      [0, k_y, k_theta, 0]])

        delta = np.array([[delta_x],
                          [delta_y],
                          [delta_orientation],
                          [delta_v]])
        u = K @ delta
        
        
        #print("u[1] is ", u[1])
        temp_u = u[1] % (2 * np.pi)
        if abs(temp_u) < 0.001:
            steer = 0 
        elif temp_u > 0:
            if temp_u < np.pi:
                steer = temp_u
            else:
                steer = temp_u - 2 * np.pi
        else:
            if temp_u > -np.pi:
                steer = temp_u
            else:
                steer = 2 * np.pi + temp_u
        '''
        if abs(u[1]) < 0.001:
            steer = 0 
        elif u[1] > np.pi:
            steer = u[1] % (np.pi)  
        elif u[1] < -np.pi:
            steer = -(u[1] % (np.pi))
        else:
            steer = u[1]
        if u[1] > 1:
            steer = 1
        elif u[1] < -1:
            steer = -1
        else: 
            steer = np.float(u[1])
        '''
        
        # control.steer = 1 if u[1] > 1 else -1 if u[1] < -1 else np.float(u[1])
    
        brake = self.min_radius / self.findCircle(self.new_waypoints[1:4])

        # print("Reach Customized Agent")
        control = carla.VehicleControl()
        control.throttle = 0.48
        control.steer = np.float(steer / np.pi) 
        #print("steer is: ", control.steer)
        if (control.steer > 0):
            print("turning right~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if (self.findCircle(self.new_waypoints[1:4]) == 0):
            
            control.brake = 0
        else:
            # control.brake = self.min_radius / self.findCircle(waypoints[1:4])
            control.brake = brake
        # print("min_radius is: ", self.min_radius)
        # print("findCircle is: ", self.findCircle(waypoints[1:4]))
        #print("brake is: ", control.brake)
        return control
        