#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Point, Pose2D, Quaternion, Transform, TransformStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int16, String, ColorRGBA
from tf2_msgs.msg import TFMessage
import tf
import numpy as np

from navigator import Mode as NavigatorMode
from final_project.msg import Cargo, DetectedObject, DetectedObjectList, PointOfInterest, PointOfInterestList

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
    POI_DROPLINE = 105
    POI_BBOX = 106
    GOAL = 107
    GOAL_DROPLINE = 108
    CARGO = 109
    NAV_MODE_TEXT = 110
    # IDs 200-299 reserved for bounding box text markers
    BBOX_TEXT_START = 200
    # IDs 300-399 reserved for zone markers
    ZONE_START = 300
    # IDs 400-499 reserved for cargo text
    CARGO_TEXT_START = 400


COLOR_DICT = {
    'stop_sign': ColorRGBA(0.3, 0.3, 0.3, 1.0),
    'apple'    : ColorRGBA(0.8, 0.2, 0.2, 1.0),
    'banana'   : ColorRGBA(0.9, 0.9, 0.1, 1.0),
    'broccoli' : ColorRGBA(0.3, 0.6, 0.3, 1.0),
    'unknown'  : ColorRGBA(1.0, 1.0, 1.0, 1.0)
}


## Markers
class PoseArrowMarker(Marker):
    def __init__(self, marker_id, transform):
        super(PoseArrowMarker, self).__init__()
        self.header.frame_id = "odom" # Following https://github.com/StanfordASL/AA274_SECTION/blob/master/s4/code/self_pub.py
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
        self.header.frame_id = "odom"
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
        self.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)


class GoalMarker(Marker):

    step = 0
    CLEARANCE = 1.0

    def __init__(self, marker_id, pose2d):
        super(GoalMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.ARROW
        self.action = 0
        self.scale.x = 0.0 # Shaft diameter
        self.scale.y = 0.1 # Head diameter
        self.scale.z = 0.5 # Head length if nonzero
        self.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)

        pt2 = Point(pose2d.x, pose2d.y, GoalMarker.CLEARANCE    +0.03*np.sin(GoalMarker.step))
        pt1 = Point(pose2d.x, pose2d.y, GoalMarker.CLEARANCE+0.1+0.03*np.sin(GoalMarker.step))
        self.points.extend([pt1, pt2])
        GoalMarker.step += 0.2


class GoalDroplineMarker(Marker):

    def __init__(self, marker_id, pose2d):
        super(GoalDroplineMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.scale.x = 0.002
        self.color = ColorRGBA(0.8, 0.8, 0.8, 1.0)

        pt1 = Point(pose2d.x, pose2d.y, GoalMarker.CLEARANCE+0.05*np.sin(GoalMarker.step))
        pt2 = Point(pose2d.x, pose2d.y, 0.0)
        self.points.extend([pt1, pt2])


class FrustumMarker(Marker):
    def __init__(self, marker_id, cam_tf, theta, detected=False):
        super(FrustumMarker, self).__init__()
        # https://answers.ros.org/question/314664/rotation-angle-in-pose-orientation/

        CD = 2.0 # Cull distance. Actual range is 300 meters
        FOV = 1.3962634
        alpha = FOV/2.0

        self.header.frame_id = "odom"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.action = 0

        self.scale.x = 0.002

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

        if detected:
            self.color = ColorRGBA(0.8, 1.0, 0.8, 1.0)
        else:
            self.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)


class BboxMarker(Marker):
    def __init__(self, marker_id, cam_tf, theta, detections):
        super(BboxMarker, self).__init__()

        GREEN_THRESH = 0.875

        FOV = 1.3962634
        alpha = FOV/2.0
        HPIX = 300 # num of pixels across
        VPIX = HPIX
        apix = HPIX/2.0 # num of pixels in part of screen subtended by alpha
                        # Cam is 300 pixels across. so 150 on each half.

        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.action = 0

        self.scale.x = 0.002

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

                a_cen = alpha / apix * (apix-xcen)  # Angle diff between theta and xcen
                a_left = alpha / apix * (apix-xmin)
                a_right = alpha / apix * (apix-xmax)
                a_top = alpha / apix * (apix-ymin)
                a_bottom = alpha / apix * (apix-ymax)

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

                points = [
                    pt_cam, pt_cen,
                    pt_tl, pt_bl,
                    pt_bl, pt_br,
                    pt_br, pt_tr,
                    pt_tr, pt_tl
                ]
                self.points.extend(points)
                if detection.confidence > GREEN_THRESH:
                    color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
                else:
                    color = ColorRGBA(0.9, 0.9, 0.0, 1.0)
                colors = [color for _ in range(len(points))]
                self.colors.extend(colors)


class BboxTextMarker(Marker):
    def __init__(self, marker_id, cam_tf, theta, detection):
        super(BboxTextMarker, self).__init__()

        FOV = 1.3962634
        alpha = FOV/2.0
        HPIX = 300 # num of pixels across
        VPIX = HPIX
        apix = HPIX/2.0 # num of pixels in part of screen subtended by alpha
                        # Cam is 300 pixels across. so 150 on each half.

        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.TEXT_VIEW_FACING
        self.action = 0

        self.scale = Vector3(0.05, 0.05, 0.05)
        self.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)

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

            a_cen = alpha / apix * (apix-xcen)  # Angle diff between theta and xcen
            # a_left = alpha / apix * (apix-xmin)
            # a_right = alpha / apix * (apix-xmax)
            a_top = alpha / apix * (apix-ymin)
            # a_bottom = alpha / apix * (apix-ymax)

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
            self.text = "{}\n{:.2f}".format(detection.name, detection.confidence)


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

        for poi in poilist.pois:
            pos = poi.position
            name = poi.name
            point = Point(pos.x, pos.y, pos.z)

            self.points.append(point)
            if name not in COLOR_DICT:
                color = COLOR_DICT['unknown']
            else:
                color = COLOR_DICT[name]
            self.colors.append(color)


class PoiBboxMarker(Marker):
    def __init__(self, marker_id, poilist):
        super(PoiBboxMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.action = 0
        self.scale.x = 0.002
        self.color = ColorRGBA(1.0, 0.2, 1.0, 1.0)

        for poi in poilist.pois:
            xmin, ymin, zmin = poi.bounds_min.x, poi.bounds_min.y, poi.bounds_min.z
            xmax, ymax, zmax = poi.bounds_max.x, poi.bounds_max.y, poi.bounds_max.z

            pt_tl = Point(xmin, ymin, zmax)
            pt_bl = Point(xmin, ymin, zmin)
            pt_tr = Point(xmax, ymax, zmax)
            pt_br = Point(xmax, ymax, zmin)

            self.points.extend([
                pt_tl, pt_bl,
                pt_bl, pt_br,
                pt_br, pt_tr,
                pt_tr, pt_tl
            ])


class PoiDroplineMarker(Marker):
    def __init__(self, marker_id, poilist):
        super(PoiDroplineMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.action = 0

        self.scale.x = 0.002
        self.color = ColorRGBA(0.3, 0.3, 0.3, 1.0)

        for poi in poilist.pois:
            pos = poi.position
            p1 = Point(pos.x, pos.y, pos.z)
            p2 = Point(pos.x, pos.y, 0.0)
            self.points.append(p1)
            self.points.append(p2)


class ZoneMarker(Marker):
    def __init__(self, marker_id, poi):
        super(ZoneMarker, self).__init__()
        self.header.frame_id = "map"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.CYLINDER

        if poi is None:
            self.action = 2 # Delete
        else:
            self.action = 0
            self.scale = Vector3(0.25, 0.25, 0.01)
            self.pose.position = copy.deepcopy(poi.position)
            name = poi.name
            if name not in COLOR_DICT:
                color = COLOR_DICT['unknown']
            else:
                color = COLOR_DICT[name]
            self.color = color


class NavModeMarker(Marker):
    def __init__(self, marker_id, transform, text):
        super(NavModeMarker, self).__init__()
        self.header.frame_id = "odom"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.TEXT_VIEW_FACING

        self.scale = Vector3(0.05, 0.05, 0.05)
        self.color = ColorRGBA(0.3, 0.3, 0.8, 1.0)
        self.pose.position.x = transform.translation.x
        self.pose.position.y = transform.translation.y
        self.pose.position.z = 0.3
        self.text = text


class CargoMarker(Marker):
    def __init__(self, marker_id, transform, theta, cargolist):
        super(CargoMarker, self).__init__()
        self.header.frame_id = "odom"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.LINE_LIST
        self.scale.x = 0.002
        self.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        # No way to access rviz's orbital camera, so we use text as a hack.

        # Draw the circle.
        R = 0.15
        cen_x = transform.translation.x
        cen_y = transform.translation.y
        cen_z = 0.15

        # Circle
        self.draw_horizontal_circle(cen_x, cen_y, cen_z, R)
        # Radials

        for idx in range(len(cargolist.cargo)):
            if idx == 0:
                angle = theta+np.pi
            elif idx == 1:
                angle = theta+np.pi-0.8
            elif idx == 2:
                angle = theta+np.pi+0.8
            else:
                angle = 0 # This shouldn't happen.

            self.draw_radial(cen_x, cen_y, cen_z, angle, R)
            self.draw_vertical_circle(cen_x+R*np.cos(angle), cen_y+R*np.sin(angle), cen_z, 0.05)

        """
        self.draw_radial(cen_x, cen_y, cen_z, theta+np.pi, R, color)
        self.draw_radial(cen_x, cen_y, cen_z, theta+np.pi-0.8, R, color)
        self.draw_radial(cen_x, cen_y, cen_z, theta+np.pi+0.8, R, color)

        self.draw_vertical_circle(cen_x+R*np.cos(theta+np.pi), cen_y+R*np.sin(theta+np.pi), cen_z, 0.05, color)
        self.draw_vertical_circle(cen_x+R*np.cos(theta+np.pi-0.8), cen_y+R*np.sin(theta+np.pi-0.8), cen_z, 0.05, color)
        self.draw_vertical_circle(cen_x+R*np.cos(theta+np.pi+0.8), cen_y+R*np.sin(theta+np.pi+0.8), cen_z, 0.05, color)
        """

    def draw_horizontal_circle(self, cen_x, cen_y, cen_z, R, n=32):
        for i in range(n):
            # if i % 2 == 0:
            #     continue
            # Horizontal
            pt1 = Point(cen_x+R*np.cos(2.0*np.pi/n*i),     cen_y+R*np.sin(2.0*np.pi/n*i),     cen_z)
            pt2 = Point(cen_x+R*np.cos(2.0*np.pi/n*(i+1)), cen_y+R*np.sin(2.0*np.pi/n*(i+1)), cen_z)
            # Vertical
            # pt1 = Point(cen_x, cen_y+R*np.sin(2.0*np.pi/n*i),     cen_z+R*np.cos(2.0*np.pi/n*i))
            # pt2 = Point(cen_x, cen_y+R*np.sin(2.0*np.pi/n*(i+1)), cen_z+R*np.cos(2.0*np.pi/n*(i+1)))
            self.points.extend([pt1, pt2])
            # self.colors.extend([color])

    def draw_vertical_circle(self, cen_x, cen_y, cen_z, R, n=16):
        for i in range(n):
            if i % 2 == 0:
                continue
            pt1 = Point(cen_x, cen_y+R*np.sin(2.0*np.pi/n*i),     cen_z+R*np.cos(2.0*np.pi/n*i))
            pt2 = Point(cen_x, cen_y+R*np.sin(2.0*np.pi/n*(i+1)), cen_z+R*np.cos(2.0*np.pi/n*(i+1)))
            self.points.extend([pt1, pt2])
            # self.colors.extend([color])


    def draw_radial(self, cen_x, cen_y, cen_z, theta, R):
        pt1 = Point(cen_x+R*np.cos(theta),   cen_y+R*np.sin(theta),   cen_z+0.05)
        pt2 = Point(cen_x+R*np.cos(theta),   cen_y+R*np.sin(theta),   cen_z+R)
        pt3 = Point(cen_x+2*R*np.cos(theta), cen_y+2*R*np.sin(theta), cen_z+2*R)
        self.points.extend([
            pt1, pt2,
            pt2, pt3
        ])
        # self.colors.extend([color, color])


class CargoTextMarker(Marker):
    def __init__(self, marker_id, transform, theta, cargolist, idx):
        super(CargoTextMarker, self).__init__()
        self.header.frame_id = "odom"
        self.header.stamp = rospy.Time()
        self.ns = MARKER_NAMESPACE
        self.id = marker_id
        self.type = self.TEXT_VIEW_FACING
        self.scale = Vector3(0.05, 0.05, 0.05)

        if idx is None:
            self.action = 2 # Delete
        else:
            self.action = 0

            name = cargolist.cargo[idx]
            if name in COLOR_DICT:
                color = COLOR_DICT[name]
            else:
                color = COLOR_DICT['unknown']
            self.text = name
            self.color = color

            # Draw the circle.
            R = 0.15
            cen_x = transform.translation.x
            cen_y = transform.translation.y
            cen_z = 0.15

            if idx == 0:
                angle = theta+np.pi
            elif idx == 1:
                angle = theta+np.pi-0.8
            elif idx == 2:
                angle = theta+np.pi+0.8
            else:
                angle = 0 # This shouldn't happen.

            self.pose.position.x = cen_x+2*R*np.cos(angle)
            self.pose.position.y = cen_y+2*R*np.sin(angle)
            self.pose.position.z = cen_z+2*R+0.02




class Visualizer(object):
    """Node responsible for rviz marker message dispatches to satisfy 'command center' requirement.
    If performance becomes a problem, cache the markers.
    """

    def __init__(self):
        rospy.init_node('turtlebot_visualizer', anonymous=True)

        # State
        self.current_pose = None    # Pose of base frame
        self.theta = 0
        self.camera_tf = None       # Transform from base_frame to base_camera
        self.nav_goal = None
        self.nav_mode = None
        self.cargo = None


        # Detection and bboxes
        self.last_detection_time = None
        self.max_simul_bboxes = 0
        self.detections = DetectedObjectList()
        self.max_simul_zones = 0
        self.max_simul_cargo = 3

        self.tf_listener = tf.TransformListener()

        # Publishers
        self.current_pose_pub       = rospy.Publisher('robot/vis/pose/current', Marker, queue_size=10)
        self.current_footprint_pub  = rospy.Publisher('robot/vis/footprint', Marker, queue_size=10)
        self.goal_pub               = rospy.Publisher('robot/vis/goal', Marker, queue_size=10)
        self.goal_dropline_pub      = rospy.Publisher('robot/vis/goal', Marker, queue_size=10)
        self.nav_mode_pub           = rospy.Publisher('robot/vis/nav_mode', Marker, queue_size=10)
        self.frustum_pub            = rospy.Publisher('robot/vis/frustum', Marker, queue_size=10)
        self.bbox_pub               = rospy.Publisher('robot/vis/bboxes', Marker, queue_size=10)
        self.bbox_text_pub          = rospy.Publisher('robot/vis/bboxes', Marker, queue_size=10)
        self.poi_pub                = rospy.Publisher('robot/vis/poi', Marker, queue_size=10)
        self.poi_dropline_pub       = rospy.Publisher('robot/vis/poi', Marker, queue_size=10)
        self.poi_bbox_pub           = rospy.Publisher('robot/vis/poi', Marker, queue_size=10)
        self.zones_pub              = rospy.Publisher('robot/vis/zones', Marker, queue_size=10)
        self.cargo_pub              = rospy.Publisher('robot/vis/cargo', Marker, queue_size=10)

        # Subscribers
        self.detector_sub = rospy.Subscriber('/detector/objects', DetectedObjectList, self.detection_cb)
        self.poi_sub      = rospy.Subscriber('/robot/poi', PointOfInterestList, self.poi_cb)
        self.zone_sub     = rospy.Subscriber('/robot/zones', PointOfInterestList, self.zone_cb)
        self.goal_sub     = rospy.Subscriber('/cmd_nav', Pose2D, self.goal_cb)
        self.nav_mode_sub = rospy.Subscriber('/nav_mode', Int16, self.nav_mode_cb)
        self.cargo_sub    = rospy.Subscriber('/robot/cargo', Cargo, self.cargo_cb)


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

    def nav_mode_cb(self, int16):
        for member in NavigatorMode:
            if member.value == int16.data:
                name = member.name
                break
        self.nav_mode = name


    def goal_cb(self, pose2d):
        self.nav_goal = pose2d


    def poi_cb(self, pointofinterestlist):
        marker = PoiMarker(RobotMarkerId.POI, pointofinterestlist)
        self.poi_pub.publish(marker)
        marker = PoiDroplineMarker(RobotMarkerId.POI_DROPLINE, pointofinterestlist)
        self.poi_dropline_pub.publish(marker)
        marker = PoiBboxMarker(RobotMarkerId.POI_BBOX, pointofinterestlist)
        self.poi_bbox_pub.publish(marker)

    
    def zone_cb(self, pointofinterestlist):
        zones = pointofinterestlist.pois
        n = len(zones)
        self.max_simul_zones = max(self.max_simul_zones, n)
        for idx in range(self.max_simul_zones):
            if idx < n:
                zone = zones[idx]
                marker = ZoneMarker(RobotMarkerId.ZONE_START+idx, zone)
            else:
                marker = ZoneMarker(RobotMarkerId.ZONE_START+idx, None)
            self.zones_pub.publish(marker)


    def cargo_cb(self, cargo):
        self.cargo = cargo


    def publish_pose_marker(self):
        marker = PoseArrowMarker(RobotMarkerId.CURRENT_POSE, self.current_pose)
        self.current_pose_pub.publish(marker)


    def publish_footprint_marker(self):
        marker = FootprintMarker(RobotMarkerId.CURRENT_FOOTPRINT, self.current_pose)
        self.current_footprint_pub.publish(marker)


    def publish_nav_mode_marker(self):
        if self.nav_mode is not None:
            marker = NavModeMarker(RobotMarkerId.NAV_MODE_TEXT, self.current_pose, "nav:" + self.nav_mode)
            self.nav_mode_pub.publish(marker)


    def publish_goal_marker(self):
        if self.nav_goal is not None:
            marker = GoalMarker(RobotMarkerId.GOAL, self.nav_goal)
            self.goal_pub.publish(marker)
            marker = GoalDroplineMarker(RobotMarkerId.GOAL_DROPLINE, self.nav_goal)
            self.goal_dropline_pub.publish(marker)


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

    
    def publish_cargo_marker(self):
        if self.cargo is not None:
            marker = CargoMarker(RobotMarkerId.CARGO, self.current_pose, self.theta, self.cargo)
            self.cargo_pub.publish(marker)
            n = len(self.cargo.cargo)
            self.max_simul_cargo = max(self.max_simul_cargo, n)
            for idx in range(self.max_simul_cargo):
                if idx < n:
                    # cargo = self.cargo.cargo[idx]
                    marker = CargoTextMarker(RobotMarkerId.CARGO_TEXT_START+idx, self.current_pose, self.theta, self.cargo, idx)
                else:
                    marker = CargoTextMarker(RobotMarkerId.CARGO_TEXT_START+idx, self.current_pose, self.theta, self.cargo, None)
                self.cargo_pub.publish(marker)
 

    def shutdown_callback(self):
        pass # TODO: Delete all markers maybe


    def run(self):
        print("Visualizer node started...")
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.update_current_pose()
            self.get_camera_tf()
            if self.current_pose is not None:
                self.publish_pose_marker()         # /robot/vis/pose/current
                self.publish_footprint_marker()    # /robot/vis/footprint
                self.publish_frustum_marker()      # /robot/vis/frustum
                self.publish_bboxes()              # /robot/vis/bboxes
                self.publish_goal_marker()         # /robot/vis/goal
                self.publish_nav_mode_marker()
                self.publish_cargo_marker()

            rate.sleep()


if __name__ == '__main__':
    node = Visualizer()
    rospy.on_shutdown(node.shutdown_callback)
    node.run()
