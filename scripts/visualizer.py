#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import tf
import numpy as np

from enum import IntEnum

MARKER_NAMESPACE = "robot"


## Marker IDs
class RobotMarkerId(IntEnum):
    CURRENT_POSE = 100 # Don't use zero as that's already used by the 2d nav goal or something.
    CURRENT_FOOTPRINT = 101


## Markers
class PoseArrowMarker(Marker):
    def __init__(self, marker_id, pose2d):
        super(PoseArrowMarker, self).__init__()
        self.header.frame_id = "map" # Following https://github.com/StanfordASL/AA274_SECTION/blob/master/s4/code/self_pub.py
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.ARROW
        self.action = 0
        self.pose.position.x = pose2d.x
        self.pose.position.y = pose2d.y
        self.pose.position.z = 0.0
        self.pose.orientation.x = 0.0 # https://answers.ros.org/question/314664/rotation-angle-in-pose-orientation/
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = np.sin(pose2d.theta / 2.0)
        self.pose.orientation.w = np.cos(pose2d.theta / 2.0)
        self.scale.x = 0.33
        self.scale.y = 0.01
        self.scale.z = 0.01
        self.color.a = 1.0
        self.color.r = 1.0
        self.color.g = 0.0
        self.color.b = 0.0


class FootprintMarker(Marker):
    def __init__(self, marker_id, pose2d):
        super(FootprintMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.CYLINDER
        self.action = 0
        self.pose.position.x = pose2d.x
        self.pose.position.y = pose2d.y
        self.pose.position.z = 0.0
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = np.sin(pose2d.theta / 2.0)
        self.pose.orientation.w = np.cos(pose2d.theta / 2.0)
        self.scale.x = 0.16 # Wheel separation = 0.16 meters
        self.scale.y = 0.16
        self.scale.z = 0.02
        self.color.a = 1.0
        self.color.r = 1.0
        self.color.g = 1.0
        self.color.b = 0.0


class Visualizer(object):
    """Node responsible for rviz marker message dispatches to satisfy 'command center' requirement.
    """

    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)

        self.current_pose = Pose2D()
        # Publishers
        self.current_pose_pub = rospy.Publisher('robot/vis/pose/current', Marker, queue_size=10)
        self.current_footprint_pub = rospy.Publisher('robot/vis/footprint', Marker, queue_size=10)
        # Subscribers
        rospy.Subscriber('/robot/pose/current', Pose2D, self.update_current_pose_cb)


    def update_current_pose_cb(self, pose2d):
        self.current_pose.x = pose2d.x
        self.current_pose.y = pose2d.y
        self.current_pose.theta = pose2d.theta


    def publish_current_pose(self):
        marker = PoseArrowMarker(RobotMarkerId.CURRENT_POSE, self.current_pose)
        self.current_pose_pub.publish(marker)


    def publish_current_footprint(self):
        marker = FootprintMarker(RobotMarkerId.CURRENT_FOOTPRINT, self.current_pose)
        self.current_footprint_pub.publish(marker)


    def shutdown_callback(self):
        pass # TODO: Delete all markers maybe


    def run(self):
        print("Visualizer node started...")
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # rospy.loginfo(self.current_pose.theta)
            self.publish_current_pose() # /robot/vis/pose/current
            self.publish_current_footprint() # /robot/vis/footprint
            rate.sleep()



if __name__ == '__main__':
    node = Visualizer()
    rospy.on_shutdown(node.shutdown_callback)
    node.run()
