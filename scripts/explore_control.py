#!/usr/bin/env python

from enum import Enum

import rospy
from final_project.msg import DetectedObject
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


class Explorer:
    def __init__(self):
        rospy.init_node('explore_node', anonymous=True)
        self.explore_queue = Queue.Queue()
        #self.explore_list = []

        self.mode = Mode.IDLE
        
        self.nav_goal_publisher = rospy.Publisher("/cmd_nav", Pose2D, queue_size=1)
        rospy.Subscriber("/nav_mode", Int16, self.nav_mode_callback)
        time.sleep(1) #Break time for the publisher to allow subsciber to contact/Establish connection

    
    def nav_mode_callback (self, data):
        print ("Mode change: ", Mode(data.data))
        self.mode = Mode(data.data)

    def explore_load_goals (self, filename):
        fileObj = open(filename, "r+")
        fileObj.readline() #Get rid of the first line (x, y, theta)

        line = fileObj.readline()
        while line != '':
            spl = line.split(", ")
            point_tuple = (float(spl[0]), float(spl[1]), float(spl[2]))
            #print("Loaded: {0}".format(str(point_tuple)))
            self.explore_queue.put(point_tuple)
            line = fileObj.readline()
    
    def send_nav_command(self, point_tuple):
        nav_g_msg = Pose2D()
        nav_g_msg.x = point_tuple[0]
        nav_g_msg.y = point_tuple[1]
        nav_g_msg.theta = point_tuple[2]

        print ("Sent Command: ", str(point_tuple))
        self.nav_goal_publisher.publish(nav_g_msg)
    
    def control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.explore_queue.empty():
                return
            
            if self.mode == Mode.IDLE or self.mode == Mode.PARK:
                next_point = self.explore_queue.get()
                self.send_nav_command(next_point)
                self.mode = Mode.NAV
            
            rate.sleep()



def main():
    expl = Explorer()
    expl.explore_load_goals("/home/mason/catkin_ws/src/AA274A-Final-Project/scripts/points.txt")
    expl.control_loop()

    

if __name__ == "__main__":
    main()

