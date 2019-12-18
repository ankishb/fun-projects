import rospy
import random
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray as F

rospy.init_node("Arm")

class Arm():
	def __init__(self):
		self._pub = rospy.Publisher("/arm_controller/command",F,queue_size=1)
	def get_joint_angles(self):
		joints = rospy.wait_for_message("/joint_states",JointState,timeout=1)
		_joints = list(joints.position)
		self._rearrange_joints(_joints)
		return tuple(self._joints)
	def _rearrange_joints(self,_joints):
		self._joints = list(_joints)
		pos1 = _joints[0];pos2 = _joints[2]
		self._joints[0] = pos2;self._joints[2] = pos1
		self._joints = [0] + self._joints[:5]
	def set_joint_positions(self,positions):
		positions = positions[1:]
		data = F()
		data.data = positions +2*[0]
		self._pub.publish(data)
	def go_home(self):
		# self.set_joint_positions(np.random.rand(5).tolist())
		self.set_joint_positions(5*[0])
# arm = Arm()
# while not rospy.is_shutdown():
# 	print(arm.get_joint_angles())
# 	# arm.set_joint_positions(5*[-0.])
# 	pass