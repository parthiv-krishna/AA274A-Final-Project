#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Point, Quaternion, Twist, Transform, TransformStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import tf
import numpy as np

import copy
from enum import IntEnum

MARKER_NAMESPACE = "robot"

## Marker IDs
class RobotMarkerId(IntEnum):
    CURRENT_POSE = 100 # Don't use zero as that's already used by the 2d nav goal.
    CURRENT_FOOTPRINT = 101
    FRUSTUM = 102



## Markers
class PoseArrowMarker(Marker):
    def __init__(self, marker_id, transform):
        super(PoseArrowMarker, self).__init__()
        self.header.frame_id = "map" # Following https://github.com/StanfordASL/AA274_SECTION/blob/master/s4/code/self_pub.py
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.ARROW
        self.action = 0
        self.pose.position = copy.deepcopy(transform.translation)
        self.pose.orientation = copy.deepcopy(transform.rotation)
        self.scale.x = 0.33
        self.scale.y = 0.01
        self.scale.z = 0.01
        self.color.a = 1.0
        self.color.r = 1.0
        self.color.g = 0.0
        self.color.b = 0.0


class FootprintMarker(Marker):
    def __init__(self, marker_id, transform):
        super(FootprintMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.CYLINDER
        self.action = 0
        self.pose.position = copy.deepcopy(transform.translation)
        self.pose.orientation = copy.deepcopy(transform.rotation)
        self.scale.x = 0.16 # Wheel separation = 0.16 meters
        self.scale.y = 0.16
        self.scale.z = 0.02
        self.color.a = 1.0
        self.color.r = 1.0
        self.color.g = 1.0
        self.color.b = 1.0


class FrustumMarker(Marker):
    def __init__(self, marker_id, cam_tf, theta, detected=False):
        super(FrustumMarker, self).__init__()
        # https://answers.ros.org/question/314664/rotation-angle-in-pose-orientation/

        CD = 2.0 # Cull distance. Actual range is 300 meters
        FOV = 1.3962634
        alpha = FOV/2.0/np.pi

        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.action = 0

        x_cam = cam_tf.translation.x
        y_cam = cam_tf.translation.y
        z_cam = cam_tf.translation.z

        pt_cam = Point(x_cam, y_cam, z_cam)
        pt_fwd = Point(x_cam + CD*np.cos(theta), y_cam + CD*np.sin(theta), z_cam)

        """
        pt_left  = Point(x_cam + CD*np.cos(theta+alpha), y_cam + CD*np.sin(theta+alpha), z_cam)
        pt_right = Point(x_cam + CD*np.cos(theta-alpha), y_cam + CD*np.sin(theta-alpha), z_cam)
        pt_top   = Point(x_cam + CD*np.cos(theta), y_cam + CD*np.sin(theta), z_cam + CD*np.sin(alpha))
        pt_btm   = Point(x_cam + CD*np.cos(theta), y_cam + CD*np.sin(theta), z_cam + CD*np.sin(-alpha))
        """
        pt_tl  = Point(x_cam + CD*np.cos(theta+alpha), y_cam + CD*np.sin(theta+alpha), z_cam + CD*np.sin(alpha))
        pt_bl  = Point(x_cam + CD*np.cos(theta+alpha), y_cam + CD*np.sin(theta+alpha), z_cam + CD*np.sin(-alpha))
        pt_tr  = Point(x_cam + CD*np.cos(theta-alpha), y_cam + CD*np.sin(theta-alpha), z_cam + CD*np.sin(alpha))
        pt_br  = Point(x_cam + CD*np.cos(theta-alpha), y_cam + CD*np.sin(theta-alpha), z_cam + CD*np.sin(-alpha))

        self.points.extend([
            pt_cam, pt_fwd,
            pt_cam, pt_tl,
            pt_cam, pt_bl,
            pt_cam, pt_tr,
            pt_cam, pt_br,
            pt_tl, pt_tr,
            pt_bl, pt_br,
            pt_tl, pt_bl,
            pt_tr, pt_br
        ])

        self.scale.x = 0.002
        self.scale.y = 0.01
        self.scale.z = 0.01
        self.color.a = 1.0
        if detected:
            self.color.r = 0.1
            self.color.g = 1.0
            self.color.b = 0.1
        else:
            self.color.r = 1.0
            self.color.g = 1.0
            self.color.b = 1.0


class VendorMarker(Marker):
    def __init__(self, marker_id):
        super(VendorMarker, self).__init__()
        # Stub


class VendorZoneMarker(Marker):
    def __init__(self, marker, id):
        super(VendorZoneMarker, self).__init__()
        # Stub


class Visualizer(object):
    """Node responsible for rviz marker message dispatches to satisfy 'command center' requirement.
    """

    def __init__(self):
        rospy.init_node('turtlebot_visualizer', anonymous=True)

        # State
        self.current_pose = None    # Pose of base frame
        self.theta = 0
        self.camera_tf = None       # Transform from base_frame to base_camera
        self.vendors = None         # Stub

        self.tf_listener = tf.TransformListener()

        # Publishers
        self.current_pose_pub       = rospy.Publisher('robot/vis/pose/current', Marker, queue_size=10)
        self.current_footprint_pub  = rospy.Publisher('robot/vis/footprint', Marker, queue_size=10)
        self.frustum_pub            = rospy.Publisher('robot/vis/frustum', Marker, queue_size=10)

        # Subscribers
        # None


    def update_current_pose(self):
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
            if self.current_pose is None:
                self.current_pose = Transform()
            self.current_pose.translation = Vector3(trans[0], trans[1], trans[2])
            self.current_pose.rotation = Quaternion(rot[0], rot[1], rot[2], rot[3])
            euler = tf.transformations.euler_from_quaternion(rot)
            self.theta = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass # Try again next loop


    def get_camera_tf(self):
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/odom', 'base_camera', rospy.Time(0))
            if self.camera_tf is None:
                self.camera_tf = Transform()
            self.camera_tf.translation = Vector3(trans[0], trans[1], trans[2])
            self.camera_tf.rotation = Quaternion(rot[0], rot[1], rot[2], rot[3])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass # Try again next loop


    def publish_pose_marker(self):
        marker = PoseArrowMarker(RobotMarkerId.CURRENT_POSE, self.current_pose)
        self.current_pose_pub.publish(marker)


    def publish_footprint_marker(self):
        marker = FootprintMarker(RobotMarkerId.CURRENT_FOOTPRINT, self.current_pose)
        self.current_footprint_pub.publish(marker)


    def publish_frustum_marker(self):
        marker = FrustumMarker(RobotMarkerId.FRUSTUM, self.camera_tf, self.theta)
        self.frustum_pub.publish(marker)


    def shutdown_callback(self):
        pass # TODO: Delete all markers maybe


    def run(self):
        print("Visualizer node started...")
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # rospy.loginfo(self.current_pose.theta)

            self.update_current_pose()
            self.get_camera_tf()
            if self.current_pose is not None:
                self.publish_pose_marker()         # /robot/vis/pose/current
                self.publish_footprint_marker()    # /robot/vis/footprint
                self.publish_frustum_marker()      # /robot/vis/frustum

            rate.sleep()


if __name__ == '__main__':
    node = Visualizer()
    rospy.on_shutdown(node.shutdown_callback)
    node.run()
