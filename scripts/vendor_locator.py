#!/usr/bin/env python

import rospy
from final_project.msg import DetectedObject, DetectedObjectList
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion

class VendorLocator:
    """Locates vendors and saves their locations"""

    pub = None
    last_pos = (0, 0, 0)
    last_pose_and_quat = None
    marker_ids = {"broccoli": (0,1), "banana":(2,3), "apple":(4,5)}

    def __init__(self):
        rospy.init_node('map_inflator', anonymous=True)

        rospy.Subscriber('/detector/objects', DetectedObjectList, self.object_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        self.pub = rospy.Publisher('/vendor_locations/banana1', Marker, queue_size=10)
        
    def object_callback(self,msg):
        for obj in msg.ob_msgs:
            print("received message ", obj.name, obj.distance)
            if obj.distance < 0.5 and obj.name in ["broccoli", "banana", "apple"]:
                marker = Marker()
    
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time()

                # IMPORTANT: If you're creating multiple markers, 
                #            each need to have a separate marker ID.
                marker.id = self.marker_ids[obj.name][0]

                marker.type = 2 # sphere

                marker.pose.position = self.last_pose_and_quat.pose.pose.position
                marker.pose.orientation = self.last_pose_and_quat.pose.pose.orientation
                
                
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                marker.color.a = 1.0 # Don't forget to set the alpha!
                marker.color.r = 1.0 if obj.name != "broccoli" else 0.0
                marker.color.g = 1.0 if obj.name != "apple" else 0.0
                marker.color.b = 0.0

                self.pub.publish(marker)
        
    def odom_callback(self,msg):
        pose = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        quat_list = [quat.x, quat.y, quat.z, quat.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        self.last_pos = (pose.x, pose.y, yaw) 
        self.last_pose_and_quat = msg

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    locator = VendorLocator()
    locator.run()
