#!/usr/bin/env python

from enum import Enum

import rospy
from final_project.msg import PointOfInterestList, Order
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String, Int16
import tf
import Queue
import time


class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    NAV = 4

class OrderHandler:

    STOP_TIME = 5.0
    zones = {}

    def __init__(self):
        rospy.init_node('order_handler', anonymous=True)
        self.waypoint_queue = Queue.Queue()        
        self.mode = Mode.IDLE
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        self.next_point = None

        self.nav_goal_publisher = rospy.Publisher("/cmd_nav", Pose2D, queue_size=1)

        self.trans_listener = tf.TransformListener()     
        
        rospy.Subscriber("/nav_mode", Int16, self.nav_mode_callback)
        rospy.Subscriber("/robot/zones", PointOfInterestList, self.zone_callback)
        rospy.Subscriber("/order", Order, self.received_order_callback) 
        time.sleep(1) #Break time for the publisher to allow subsciber to contact/Establish connection

    def zone_callback(self, data):        
        self.zones = {}
        for poi in data.pois:
            if poi.name not in self.zones:
                self.zones[poi.name] = []
            self.zones[poi.name].append((poi.position.x, poi.position.y, 0))
    
    def received_order_callback(self, data): 
        #for vendor in data.foods:
        print("received order")
        print("foods" + str(data.foods))
        print("location\n" + str(data.location))
        
        for food in data.foods:
            print(food)
            self.waypoint_queue.put(self.zones[food][0])
            
        self.waypoint_queue.put((data.location.x, data.location.y, 0))  
    
    def nav_mode_callback(self, data):
        print ("Mode change: ", Mode(data.data))
        self.mode = Mode(data.data)

    def send_nav_command(self, point_tuple):
        nav_g_msg = Pose2D()
        nav_g_msg.x = point_tuple[0]
        nav_g_msg.y = point_tuple[1]
        nav_g_msg.theta = point_tuple[2]

        print ("Sent Command: ", str(point_tuple))
        self.nav_goal_publisher.publish(nav_g_msg)
    
    def control_loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.waypoint_queue.empty():
                #print("waypoint queue empty")
                #print(str(self.pois))
                continue                
            
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.mode = Mode.IDLE
                print e
                pass

            if self.mode == Mode.IDLE or self.mode == Mode.PARK:
                self.next_point = (self.x, self.y, self.theta) if self.next_point is None else self.next_point                
                if (self.x - self.next_point[0])**2 + (self.y - self.next_point[1])**2 < 0.25:
                    # reached point
                    self.last_point = self.next_point
                    self.next_point = self.waypoint_queue.get()
                    rospy.loginfo("Orderer reached waypoint, stopping for " + str(self.STOP_TIME) + " seconds")
                    rospy.sleep(self.STOP_TIME)
                self.send_nav_command(self.next_point)
                if (self.x - self.last_point[0])**2 + (self.y - self.last_point[1])**2 > 0.25:
                    # left last point
                    self.mode = Mode.NAV
                    rospy.loginfo("Left last point, entering NAV mode")  
            
            #else:
                #print ("Position [x, y, theta]: {0}, {1}, {2}".format(self.x, self.y, self.theta))
            rate.sleep()

def main():
    orderhandler = OrderHandler()
    orderhandler.control_loop()

if __name__ == "__main__":
    main()
