#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Point, Quaternion, Twist, Transform, TransformStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA
from tf2_msgs.msg import TFMessage
import tf
import numpy as np

from final_project.msg import DetectedObject, DetectedObjectList, PointOfInterest, PointOfInterestList

import copy
from enum import IntEnum

MARKER_NAMESPACE = "robot"

## Marker IDs
class RobotMarkerId(IntEnum):
    CURRENT_POSE = 100 # Don't use zero as that's already used by the 2d nav goal.
    CURRENT_FOOTPRINT = 101
    FRUSTUM = 102
    BBOX = 103
    POI = 104
    ZONES = 105

    # IDs 200-299 reserved for bounding box text markers
    BBOX_TEXT_START = 200


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
        alpha = FOV/2.0

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
            self.color.r = 0.8
            self.color.g = 1.0
            self.color.b = 0.8
        else:
            self.color.r = 1.0
            self.color.g = 1.0
            self.color.b = 1.0


class BboxMarker(Marker):
    def __init__(self, marker_id, cam_tf, theta, detections):
        super(BboxMarker, self).__init__()

        FOV = 1.3962634
        alpha = FOV/2.0

        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.action = 0

        self.scale.x = 0.002
        self.scale.y = 0.01
        self.scale.z = 0.01
        self.color.a = 1.0

        self.color.r = 0.1
        self.color.g = 1.0
        self.color.b = 0.1

        if detections is None:
            # print("no detection")
            self.action = 2 # Delete
        else:
            for detection in detections.ob_msgs:

                distance = detection.distance
                corners = detection.corners

                x_cam = cam_tf.translation.x
                y_cam = cam_tf.translation.y
                z_cam = cam_tf.translation.z
                pt_cam = Point(x_cam, y_cam, z_cam)

                ymin, xmin = corners[0], corners[1]
                ymax, xmax = corners[2], corners[3]
                xcen, ycen = (xmax+xmin)/2, (ymax+ymin)/2
                d = distance

                # print("{} {}".format(xcen, ycen))

                # Cam is 300 pixels across. so 150 on each half.
                # beta  = alpha / 150.0 * (150.0 - ycen)
                a_cen = alpha / 150.0 * (150.0-xcen)  # Angle diff between theta and xcen
                a_left = alpha / 150.0 * (150.0-xmin)
                a_right = alpha / 150.0 * (150.0-xmax)
                a_top = alpha / 150.0 * (150.0-ymin)
                a_bottom = alpha / 150.0 * (150.0-ymax)

                """
                pt_left = Point(x_cam + d*np.cos(theta+a_left), y_cam + d*np.sin(theta+a_left), z_cam + d*np.sin((a_top+a_bottom)/2))
                pt_right = Point(x_cam + d*np.cos(theta+a_right), y_cam + d*np.sin(theta+a_right), z_cam + d*np.sin((a_top+a_bottom)/2))
                pt_top = Point(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin(a_top))
                pt_bottom = Point(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin(a_bottom))
                """
                pt_cen = Point(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin((a_top+a_bottom)/2))
                pt_tl  = Point(x_cam + d*np.cos(theta+a_left), y_cam + d*np.sin(theta+a_left), z_cam + d*np.sin(a_top))
                pt_bl  = Point(x_cam + d*np.cos(theta+a_left), y_cam + d*np.sin(theta+a_left), z_cam + d*np.sin(a_bottom))
                pt_tr  = Point(x_cam + d*np.cos(theta+a_right), y_cam + d*np.sin(theta+a_right), z_cam + d*np.sin(a_top))
                pt_br  = Point(x_cam + d*np.cos(theta+a_right), y_cam + d*np.sin(theta+a_right), z_cam + d*np.sin(a_bottom))

                self.points.extend([
                    pt_cam, pt_cen,
                    pt_tl, pt_bl,
                    pt_bl, pt_br,
                    pt_br, pt_tr,
                    pt_tr, pt_tl
                ])


class BboxTextMarker(Marker):
    def __init__(self, marker_id, cam_tf, theta, detection):
        super(BboxTextMarker, self).__init__()

        FOV = 1.3962634
        alpha = FOV/2.0

        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.TEXT_VIEW_FACING
        self.action = 0

        self.scale.x = 0.05
        self.scale.y = 0.05
        self.scale.z = 0.05

        self.color.a = 1.0
        self.color.r = 1.0
        self.color.g = 1.0
        self.color.b = 1.0

        if detection is None:
            # print("no detection")
            self.action = 2 # Delete
        else:
            distance = detection.distance
            corners = detection.corners

            x_cam = cam_tf.translation.x
            y_cam = cam_tf.translation.y
            z_cam = cam_tf.translation.z
            pt_cam = Point(x_cam, y_cam, z_cam)

            ymin, xmin = corners[0], corners[1]
            ymax, xmax = corners[2], corners[3]
            xcen, ycen = (xmax+xmin)/2, (ymax+ymin)/2
            d = distance

            a_cen = alpha / 150.0 * (150.0-xcen)  # Angle diff between theta and xcen
            # a_left = alpha / 150.0 * (150.0-xmin)
            # a_right = alpha / 150.0 * (150.0-xmax)
            a_top = alpha / 150.0 * (150.0-ymin)
            # a_bottom = alpha / 150.0 * (150.0-ymax)

            # print("{} {}".format(xcen, ycen))
            """
            pt_cen = Point(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin((a_top+a_bottom)/2))
            pt_left = Point(x_cam + d*np.cos(theta+a_left), y_cam + d*np.sin(theta+a_left), z_cam + d*np.sin((a_top+a_bottom)/2))
            pt_right = Point(x_cam + d*np.cos(theta+a_right), y_cam + d*np.sin(theta+a_right), z_cam + d*np.sin((a_top+a_bottom)/2))
            pt_top = Point(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin(a_top))
            pt_bottom = Point(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin(a_bottom))
            """
            self.pose.position.x = x_cam + d*np.cos(theta+a_cen)
            self.pose.position.y = y_cam + d*np.sin(theta+a_cen)
            self.pose.position.z = z_cam + d*np.sin(a_top+0.05)
            self.text = detection.name


class PoiMarker(Marker):
    def __init__(self, marker_id, poilist):
        super(PoiMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.POINTS
        self.action = 0

        self.scale.x = 0.07
        self.scale.y = 0.07
        self.scale.z = 0.0

        for poi in poilist.pois:
            pos = poi.position
            name = poi.name
            point = Point(pos.x, pos.y, pos.z)

            self.points.append(point)
            if name == 'stop_sign':
                color = ColorRGBA(0.3, 0.3, 0.3, 1.0)
            elif name == 'apple':
                color = ColorRGBA(1.0, 0.2, 0.2, 1.0)
            elif name == 'banana':
                color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
            elif name == 'broccoli':
                color = ColorRGBA(0.3, 0.8, 0.3, 1.0)
            else: # Unknown
                color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            self.colors.append(color)



class ZoneMarker(Marker):
    def __init__(self, marker_id, poilist):
        super(ZoneMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.SPHERE_LIST
        self.action = 0

        self.scale.x = 0.2

        for poi in poilist.pois:
            pos = poi.position
            name = poi.name
            point = Point(pos.x, pos.y, 0.0)

            self.points.append(point)
            if name == 'stop_sign':
                color = ColorRGBA(0.3, 0.3, 0.3, 1.0)
            elif name == 'apple':
                color = ColorRGBA(1.0, 0.2, 0.2, 1.0)
            elif name == 'banana':
                color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
            elif name == 'broccoli':
                color = ColorRGBA(0.3, 0.8, 0.3, 1.0)
            else: # Unknown
                color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            self.colors.append(color)



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

        # Detection and bboxes
        self.last_detection_time = None
        self.max_simul_bboxes = 0
        self.detections = DetectedObjectList()

        self.tf_listener = tf.TransformListener()

        # Publishers
        self.current_pose_pub       = rospy.Publisher('robot/vis/pose/current', Marker, queue_size=10)
        self.current_footprint_pub  = rospy.Publisher('robot/vis/footprint', Marker, queue_size=10)
        self.frustum_pub            = rospy.Publisher('robot/vis/frustum', Marker, queue_size=10)
        self.bbox_pub               = rospy.Publisher('robot/vis/bboxes', Marker, queue_size=10)
        self.bbox_text_pub          = rospy.Publisher('robot/vis/bbox_text', Marker, queue_size=10)
        self.poi_pub                = rospy.Publisher('robot/vis/poi', Marker, queue_size=10)
        self.zones_pub              = rospy.Publisher('robot/vis/zones', Marker, queue_size=10)

        # Subscribers
        self.detector_sub = rospy.Subscriber('/detector/objects', DetectedObjectList, self.detection_cb)
        self.poi_sub      = rospy.Subscriber('/robot/poi', PointOfInterestList, self.poi_cb)
        self.zone_sub     = rospy.Subscriber('/robot/zones', PointOfInterestList, self.zone_cb)


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


    def detection_cb(self, data):
        self.last_detection_time = rospy.Time().now()
        self.detections = data


    def is_detection_stale(self, thresh=0.1):
        """Checks if detection boxes are old, because messages are not timestamped."""
        time = rospy.Time().now()
        if self.last_detection_time is None or (time - self.last_detection_time) > rospy.Duration(secs=thresh):
            return True
        else:
            return False


    def poi_cb(self, pointofinterestlist):
        marker = PoiMarker(RobotMarkerId.POI, pointofinterestlist)
        self.poi_pub.publish(marker)

    
    def zone_cb(self, pointofinterestlist):
        marker = ZoneMarker(RobotMarkerId.ZONES, pointofinterestlist)
        self.zones_pub.publish(marker)


    def publish_pose_marker(self):
        marker = PoseArrowMarker(RobotMarkerId.CURRENT_POSE, self.current_pose)
        self.current_pose_pub.publish(marker)


    def publish_footprint_marker(self):
        marker = FootprintMarker(RobotMarkerId.CURRENT_FOOTPRINT, self.current_pose)
        self.current_footprint_pub.publish(marker)


    def publish_frustum_marker(self):
        has_detections = not self.is_detection_stale()
        marker = FrustumMarker(RobotMarkerId.FRUSTUM, self.camera_tf, self.theta, has_detections)
        self.frustum_pub.publish(marker)


    def publish_bboxes(self):
        if not self.is_detection_stale():
            detections = self.detections
        else:
            detections = None
        # Publish the box
        marker = BboxMarker(RobotMarkerId.BBOX, self.camera_tf, self.theta, detections)
        self.bbox_pub.publish(marker)
        # Publish the text
        n = len(self.detections.ob_msgs)
        self.max_simul_bboxes = max(self.max_simul_bboxes, n)
        for idx in range(self.max_simul_bboxes):
            if idx < n and detections is not None:
                detection = detections.ob_msgs[idx]
                marker = BboxTextMarker(RobotMarkerId.BBOX_TEXT_START+idx, self.camera_tf, self.theta, detection)
            else:
                marker = BboxTextMarker(RobotMarkerId.BBOX_TEXT_START+idx, self.camera_tf, self.theta, None)
            self.bbox_text_pub.publish(marker)
 

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
                self.publish_bboxes()              # /robot/vis/bboxes

            rate.sleep()


if __name__ == '__main__':
    node = Visualizer()
    rospy.on_shutdown(node.shutdown_callback)
    node.run()
