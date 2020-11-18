#!/usr/bin/env python

import rospy
from final_project.msg import DetectedObject, DetectedObjectList, Vendor, VendorList
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class VendorLocator:
    """Locates vendors and saves their locations"""

    pub = None
    last_pos = (0, 0, 0)
    last_pose_and_quat = None
    marker_ids = {"broccoli": (0,1), "banana":(2,3), "apple":(4,5)}
    vendors = []

    def __init__(self):
        rospy.init_node('map_inflator', anonymous=True)

        rospy.Subscriber('/detector/objects', DetectedObjectList, self.object_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        self.pub = rospy.Publisher('/vendor_locations', VendorList, queue_size=10)
        
    def object_callback(self,msg):
        for obj in msg.ob_msgs:
            print("received message ", obj.name, obj.distance)
            if obj.distance < 0.5 and obj.name in ["broccoli", "banana", "apple", "stop sign"]:
                vendor = Vendor()
                vendor.x, vendor.y, _ = self.last_pos
                vendor.id = obj.id
                vendor.name = obj.name
                for existing_vendor in self.vendors:
                    sqdist = (existing_vendor.x - vendor.x)**2 + (existing_vendor.y - vendor.y)**2
                    if existing_vendor.id == vendor.id and sqdist < 1:
                        return
                self.vendors.append(vendor)
            
        
    def odom_callback(self,msg):
        pose = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        quat_list = [quat.x, quat.y, quat.z, quat.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        self.last_pos = (pose.x, pose.y, yaw) 
        self.last_pose_and_quat = msg

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.pub.publish(self.vendors)
            rate.sleep()
            

if __name__ == '__main__':
    locator = VendorLocator()
    locator.run()
