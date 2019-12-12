# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os,random
import time
from os.path import expanduser

import gym
import numpy as np
import pybullet
import pybullet_data
#import arm
from gym import spaces
from gym.utils import seeding

from . import bullet_client


class UrGymEnv(gym.Env):

    def __init__(self):
        self._renders = False
        self._urdf_root = None  # path to the urdf
        self._action_dim = 6  # number of joint to control
        observation_dim = self._action_dim + 6  # observation is current joint state plus goal
        self._collision_check = False  # whether to do collision check while simulation
        observation_high = np.ones(observation_dim) * 1000  # np.inf
        self._action_bound = 1
        action_high = np.array([self._action_bound] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None
        self._joint_start = 1  # this corresponds to the joint 1 of arm
        self._num_steps = 0
        self._max_episode_steps = 300
        self._threshold = 0.01  # (distance in m) if robot reaches within this distance, goal is reached
        self._collision_reward = -10  # reward corresponding to collision
        self._max_angle_change = np.deg2rad(1)
        self._action_rew_coeff = 0
        self._valid_goal = True  # only provide reachable goal or not
        self._angle_search_limits = 90
        self._stop_on_collision = False  # if collision_check is on, then whether to stop episode on collision
        self._p = None
        self._use_real_robot = False  # whether to use real robot
        self._real_robot = None
        self._sleep_after_exc = 2.50  # (sec) sleep for this time after every action, if executing on real robot
        self._reaching_rew = 1  # reward if end-effector reaches within threshold distance from goal
        self._collision = False
        self._start_orient = np.random.uniform(-1,1,size=[10,3])
        self._goal_orient = np.random.uniform(-1,1,size=[10,3])

    def _set_env(self):
        # gym.make doesnt allow to pass arguments, this is kind of workaround for that
        if self._urdf_root is None:
            self._urdf_root = os.path.join("/home/amrut/My codes/ur_description", '/home/amrut/My codes/ur_description/urdf/ur10_robot.urdf')
        if self._renders:
#            print(state_dim,action_dim,max_action)
            self._p = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -10)
        self._plane_id = self._p.loadURDF("plane.urdf",basePosition=[0,0,-1.5])
        if self._collision_check:
            self._sim_robot = self._p.loadURDF(self._urdf_root, useFixedBase=1, flags=self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        else:
            self._sim_robot = self._p.loadURDF(self._urdf_root, useFixedBase=1)

        if self._use_real_robot:
            self._real_robot = arm.Arm()
            time.sleep(self._sleep_after_exc)

    def reset(self):
        
        if self._p is None: self._set_env()
        if self._renders:
            self._p.removeAllUserDebugItems()
        self._goal = self._get_goal()
#        while np.lin
#        print(self._goal)
        # discard goal if z < 0.2
#        if self._use_real_robot:
#            while self._goal[2] < 0.2 or self._goal[0] < 0.4:
#                if self._renders:
#                    self._p.removeAllUserDebugItems()
#                self._goal = self._get_goal()
#        self._goal = [0.5,-0.25,-0.4]
        
        self._num_steps = 0
        if self._use_real_robot:
#            self._real_robot.arm.go_home()
            self._real_robot.go_home()
            time.sleep(self._sleep_after_exc)

        if self._collision_check:
            self._p.stepSimulation()
            self._p.stepSimulation()
            self._num_collision_pt = len(self._p.getContactPoints())

        self._collision = False

        
        return np.array(self._get_joints() + self._goal)

    def step(self, a):
        self._num_steps += 1
        # get the new_theta
        required_joints = np.clip(np.array(self._get_joints()) + a * self._max_angle_change, -np.pi / 2, np.pi / 2)

        if self._use_real_robot:
            # execute on real robot
            arm_joint = required_joints.tolist() + (5 - self._action_dim) * [0]
#            print("arm_joint",arm_joint)
#            self._real_robot.arm.set_joint_positions(arm_joint, plan=False, wait=False)
            self._real_robot.set_joint_positions(arm_joint)      # edited
            time.sleep(self._sleep_after_exc)
#            required_joints = self._real_robot.arm.get_joint_angles()
            required_joints = self._real_robot.get_joint_angles()      # edited
            # execute by directly writing the joint angles
            for i in range(self._action_dim):
                self._p.resetJointState(bodyUniqueId=self._sim_robot,
                                        jointIndex=i + self._joint_start ,
                                        targetValue=required_joints[i])
            dist, rew = self._cal_reward(a)

        else:
            # execute by directly writing the joint angles
            for i in range(self._action_dim):
                self._p.resetJointState(bodyUniqueId=self._sim_robot,
                                        jointIndex=i + self._joint_start,
                                        targetValue=required_joints[i])
            self._p.stepSimulation()
            dist, rew = self._cal_reward(a)
            if self._collision_check:
                self._collision = self._detect_collision()
                if self._collision:
                    rew += self._collision_reward
                    if self._stop_on_collision:
                        return np.array(self._get_joints() + self._goal), rew, True, {}

        if dist < self._threshold:
            print(dist)
            return np.array(self._get_joints() + self._goal), self._reaching_rew, True, {}
#        print(dist)
        return np.array(self._get_joints() + self._goal), rew, self._num_steps >= self._max_episode_steps, {}
#        return np.array(self._get_joints() + self._goal), rew, False, {}


    def _get_goal(self):
        
        if self._valid_goal:
            # for valid goal we will sample through joint angles
            random_joint = self._get_Goal()
#            random_joint = random.random()*(np.random.rand(self._action_dim) - 0.5) * np.radians(180)

#            random_joint = (1*np.random.rand(self._action_dim)) * np.radians(180)
            for i in range(self._action_dim):
                self._p.resetJointState(bodyUniqueId=self._sim_robot,
                                        jointIndex=i + self._joint_start,
                                        targetValue=random_joint[i])

            # the [x,y,z] reached by the co-ordinate becomes the goal wrt to world origin
            goal = self._get_position()
            
            # reset back to the zero angles
            start_joint = self._get_start()
            for i in range(self._action_dim):
                self._p.resetJointState(bodyUniqueId=self._sim_robot,
                                        jointIndex=i + self._joint_start,
                                        targetValue=start_joint[i])

        else:
            # random goals
            pos = self._p.getLinkState(self._sim_robot, self._joint_start)[-2]
            goal = [np.random.rand() - 0.5 + pos[0],
                    np.random.rand() - 0.5 + pos[1],
                    0.5 * np.random.rand() + pos[2]]
        # to see where is the goal
        if self._renders:
            self._p.addUserDebugLine(lineFromXYZ=3 * [0],
                                     lineToXYZ=goal[:3],
                                     lineColorRGB=[0, 0, 1], lineWidth=10.0, lifeTime=0)

        return list(goal)
        

    def _get_joints(self):
        
        if self._use_real_robot:
#            joint_state = self._real_robot.arm.get_joint_angles().tolist()[:self._action_dim]
            joint_state = self._real_robot.get_joint_angles()[:self._action_dim]   # edited
            joint_state = list(joint_state)                                        # edited
            # for visualization
            for i in range(self._action_dim):
                self._p.resetJointState(bodyUniqueId=self._sim_robot,
                                        jointIndex=i + self._joint_start,
                                        targetValue=joint_state[i])
        else:
            joint_state = []
            for i in range(self._action_dim):
                joint_state.append(self._p.getJointState(bodyUniqueId=self._sim_robot,
                                                         jointIndex=self._joint_start + i)[0])
        return joint_state

    def _get_position(self):
       
        pos = self._p.getLinkState(self._sim_robot, self._joint_start + self._action_dim)[-2]
        orn = self._p.getEulerFromQuaternion(self._p.getLinkState(self._sim_robot, self._joint_start + self._action_dim)[-1])

        # to see where is the goal
        if self._renders:
            self._p.addUserDebugLine(lineFromXYZ=3 * [0],
                                     lineToXYZ=pos,
                                     lineColorRGB=[1, 0, 0], lineWidth=10.0, lifeTime=0.1)
        return pos + orn

    def _cal_reward(self, a):
        pos = list(self._get_position())
        pos[:3] = np.array(pos[:3])
#        print(pos)
        pos[3:] = map(np.sin,pos[3:])
        dist = np.linalg.norm(np.array(pos[:3]) - np.array(self._goal[:3]), ord=2)
        return dist, -dist - self._action_rew_coeff * np.mean(np.abs(a))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _detect_collision(self):
        num_collision_pt = len(self._p.getContactPoints())
        if num_collision_pt > self._num_collision_pt:
            self._num_collision_pt = num_collision_pt
            return True
        if self._collision and num_collision_pt == self._num_collision_pt:
            return True
        self._num_collision_pt = num_collision_pt
        return False
    def _get_ik(self,pos):
        
        joints = self._p.calculateInverseKinematics(self._sim_robot,endEffectorLinkIndex=7,targetPosition=pos[:3],targetOrientation=self._p.getQuaternionFromEuler(pos[3:]))
        return joints
    
    def _filter_goal(self):
        goal_xpos = np.random.uniform(0.2,0.8)
        goal_ypos = np.random.uniform(-0.8,0.8)
        while np.abs(goal_ypos) < 0.2:
            goal_ypos = np.random.uniform(-0.8,0.8)
        goal_zpos = np.random.uniform(-0.8,0.4)
        goal_orn = self._goal_orient[random.randint(0,9)]
#        goal_orn =self._sim_robot.getQuaternionFromEuler(goal_orn)
        return (goal_xpos,goal_ypos,goal_zpos)+tuple(goal_orn)
    
    def _filter_start(self):
        goal_xpos = np.random.uniform(0.2,0.8)
        goal_ypos = np.random.uniform(-0.8,0.8)
        while np.abs(goal_ypos) < 0.2:
            goal_ypos = np.random.uniform(-0.8,0.8)
        goal_zpos = np.random.uniform(-0.8,0.4)
        goal_orn = self._start_orient[random.randint(0,9)]
#        goal_orn =self._sim_robot.getQuaternionFromEuler(goal_orn)
        return (goal_xpos,goal_ypos,goal_zpos)+tuple(goal_orn)
    
    def _get_start(self):
        self._start = self._get_ik(self._filter_start())
        return list(self._start)
    
    def _get_Goal(self):
        self._goal = self._get_ik(self._filter_goal())
        return list(self._goal)

    