import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray as F

rospy.init_node("Arm")

class Arm():
	def __init__(self):
		self._pub = rospy.Publisher("/arm_controller/command",F,queue_size=1)
	def get_joint_angles(self):
		joints = rospy.wait_for_message("/joint_states",JointState,timeout=0.5)
		self._joints = joints.position
		return self._joints
	def set_joint_positions(self,positions):
		data = F()
		data.data = positions +[0]
		self._pub.publish(data)

# arm = Arm()
# while not rospy.is_shutdown():
# 	print(arm.get_joint_angles())
# 	arm.set_joint_positions(5*[-0.])
# 	pass