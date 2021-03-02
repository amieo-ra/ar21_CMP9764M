#!/usr/bin/env python
# Author Willow Mandil || 01/02/2021

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import PoseStamped


class FrankaRobot(object):
	def __init__(self):
		super(FrankaRobot, self).__init__()
		moveit_commander.roscpp_initialize(sys.argv)
		rospy.init_node('FrankaRobotWorkshop', anonymous=True)
		self.robot = moveit_commander.RobotCommander()
		self.scene = moveit_commander.PlanningSceneInterface()
		self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
		self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
														moveit_msgs.msg.DisplayTrajectory,
														queue_size=20)
		self.planning_frame = self.move_group.get_planning_frame()
		self.eef_link = self.move_group.get_end_effector_link()
		self.group_names = self.robot.get_group_names()

	def get_robot_joint_state(self):
		robot_state = self.robot.get_current_state().joint_state.position
		print(robot_state)

	def get_robot_task_state(self):
		robot_ee_pose = self.move_group.get_current_pose().pose
		print(robot_ee_pose)

	def go_to_joint_state(self):
		joint_goal = [0, -pi/4, 0, -pi/2, 0, pi/3, 0]
		self.move_group.go(joint_goal, wait=True)
		self.move_group.stop() # ensures there are no residual movements

	def go_to_task_state(self):
		p = PoseStamped()
		p.header.frame_id = '/panda_link0'
		p.pose.position.x = 0.45
		p.pose.position.y = -0.25
		p.pose.position.z = 0.45
		p.pose.orientation.x = 1
		p.pose.orientation.y = 0
		p.pose.orientation.z = 0
		p.pose.orientation.w = 0
		target = self.move_group.set_pose_target(p)
		self.move_group.go(target)

	def follow_path(self):
		waypoints = []

		wpose = self.move_group.get_current_pose().pose
		wpose.position.z -= 0.1  # First move up (z)
		wpose.position.y += 0.2  # and sideways (y)
		waypoints.append(copy.deepcopy(wpose))

		wpose.position.x += 0.1  # Second move forward/backwards in (x)
		waypoints.append(copy.deepcopy(wpose))

		wpose.position.y -= 0.1  # Third move sideways (y)
		waypoints.append(copy.deepcopy(wpose))

		(plan, fraction) = self.move_group.compute_cartesian_path(
										   waypoints,   # waypoints to follow
										   0.01,        # eef_step
										   0.0)         # jump_threshold

		# display trajectory:
		display_trajectory = moveit_msgs.msg.DisplayTrajectory()
		display_trajectory.trajectory_start = self.robot.get_current_state()
		display_trajectory.trajectory.append(plan)
		self.display_trajectory_publisher.publish(display_trajectory);
		
		raw_input("press enter to execute")
		# execute trajectory:
		self.move_group.execute(plan, wait=True)


if __name__ == '__main__':
	robot = FrankaRobot()
	# robot.get_robot_joint_state()
	# robot.get_robot_task_state()
	# robot.go_to_joint_state()
	# robot.go_to_task_state()
	robot.follow_path()