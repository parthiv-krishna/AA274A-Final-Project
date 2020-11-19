#!/usr/bin/env python

import rospy
import tf
from final_project.msg import (DetectedObject, DetectedObjectList,
                                PointOfInterest, PointOfInterestList, Vendor, VendorList)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Transform, TransformStamped, Vector3
from tf.transformations import euler_from_quaternion

import numpy as np

class PoiLocator:
    """Locates points of interest and saves + publishes their locations."""

    POI_NAMES    = set(["broccoli", "banana", "apple", "stop_sign"])
    VENDOR_NAMES = set(["broccoli", "banana", "apple"])

    POI_EXCLUSION_DIST = 0.5 # meters
    ZONE_EXCLUSION_DIST = 1.0 # meters
    CONFIDENCE_THRESH = 0.825

    def __init__(self):
        rospy.init_node('poi_locator', anonymous=True)
        self.current_pose = None
        self.theta = 0

        # Used to identify closest vendor of interest.
        self.target_zone_name = None
        self.target_zone_idx = None

        # Dict[str] of [(Vector3, detection_dist)]. The key (name, idx) uniquely identifies any poi or zone.
        self.pois = {}      # Positions of stop signs, vendor signs, puddles etc.
        self.zones = {}     # Zones associated with points of interest, e.g. vendor pickup spot.

        self.tf_listener = tf.TransformListener()
        self.camera_tf = None       # Transform from base_frame to base_camera

        # Subscribers
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.detection_callback)

        # Publishers
        self.poi_pub = rospy.Publisher('/robot/poi', PointOfInterestList, queue_size=10)
        self.zone_pub     = rospy.Publisher('/robot/zones', PointOfInterestList, queue_size=10)


    def get_current_pose(self):
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


    def localize_object(self, detected_object):
        """Returns center coord of detected object in terms of /odom frame."""
        FOV = 1.3962634
        alpha = FOV/2.0
        theta = self.theta
        distance = detected_object.distance
        corners = detected_object.corners
        cam_tf = self.camera_tf

        x_cam = cam_tf.translation.x
        y_cam = cam_tf.translation.y
        z_cam = cam_tf.translation.z
        # pt_cam = Point(x_cam, y_cam, z_cam)

        ymin, xmin = corners[0], corners[1]
        ymax, xmax = corners[2], corners[3]
        xcen, ycen = (xmax+xmin)/2, (ymax+ymin)/2
        d = distance

        a_cen = alpha / 150.0 * (150.0-xcen)
        # a_left = alpha / 150.0 * (150.0-xmin)
        # a_right = alpha / 150.0 * (150.0-xmax)
        a_top = alpha / 150.0 * (150.0-ymin)
        a_bottom = alpha / 150.0 * (150.0-ymax)

        pt_cen = Vector3(x_cam + d*np.cos(theta+a_cen), y_cam + d*np.sin(theta+a_cen), z_cam + d*np.sin((a_top+a_bottom)/2))
        return pt_cen


    def euclidean_distance(self, p1, p2):
        """Return L2 distance between two Vector3 objects."""
        return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)


    # Note that right now if we pick up a false positive the algorithm as it is cannot erase it.

    def register_poi(self, detected_object):
        name = detected_object.name
        dist = detected_object.distance
        confidence = detected_object.confidence  # simple approach to avoid false positives
        if confidence > self.CONFIDENCE_THRESH and name in PoiLocator.POI_NAMES:
            pos = self.localize_object(detected_object)
            if name not in self.pois:
                self.pois[name] = []
            is_new_spot = True
            for idx in range(len(self.pois[name])): # O(n) naive comparison
                ref_pos, ref_dist = self.pois[name][idx]
                is_new_spot = (is_new_spot and
                                self.euclidean_distance(pos, ref_pos) > PoiLocator.POI_EXCLUSION_DIST)
                if not is_new_spot:
                    if dist < ref_dist:
                        self.pois[name][idx] = (pos, dist) # Refine position
                    break
            if is_new_spot:
                self.pois[name].append((pos, dist))
        

    def register_zone(self, detected_object):
        obj = detected_object
        name = obj.name
        dist = detected_object.distance
        confidence = detected_object.confidence  # simple approach to avoid false positives
        if confidence > self.CONFIDENCE_THRESH and obj.distance < 1 and obj.name in PoiLocator.VENDOR_NAMES:
            pos = self.current_pose.translation # x,y,z
            if name not in self.zones:
                self.zones[name] = []
            is_new_spot = True
            for idx in range(len(self.zones[name])): # O(n) naive comparison
                ref_pos, ref_dist = self.zones[name][idx]
                is_new_spot = (is_new_spot and
                                self.euclidean_distance(pos, ref_pos) > PoiLocator.POI_EXCLUSION_DIST)
                if not is_new_spot:
                    if dist < ref_dist:
                        self.zones[name][idx] = (pos, dist)
                    break
            if is_new_spot:
                self.zones[name].append((pos, dist))

    
    def detection_callback(self, detected_object_list):
        if self.current_pose is not None:
            for obj in detected_object_list.ob_msgs:
                # print("received message ", obj.name, obj.distance)
                self.register_poi(obj)
                self.register_zone(obj)


    def publish_poi(self):
        msg = PointOfInterestList()
        for name in self.pois:
            ls = self.pois[name]
            for idx in range(len(ls)):
                poi = PointOfInterest()
                poi.name = name
                poi.is_target = False
                poi.position = ls[idx][0]
                msg.pois.append(poi)
        self.poi_pub.publish(msg)


    def publish_zones(self):
        msg = PointOfInterestList()
        for name in self.zones:
            ls = self.zones[name]
            for idx in range(len(ls)):
                poi = PointOfInterest()
                poi.name = name
                if name == self.target_zone_name and idx == self.target_zone_idx:
                    poi.is_target = True
                else:
                    poi.is_target = False
                poi.position = ls[idx][0]
                msg.pois.append(poi)
        self.zone_pub.publish(msg)


    def shutdown_callback(self):
        pass # Do nothing.


    def run(self):
        rate = rospy.Rate(10)
        print("poi_locator node started...")
        while not rospy.is_shutdown():
            self.get_current_pose()
            self.get_camera_tf()
            if self.current_pose is not None:
                self.publish_poi()
                self.publish_zones()
            rate.sleep()


if __name__ == '__main__':
    node = PoiLocator()
    rospy.on_shutdown(node.shutdown_callback)
    node.run()
    
