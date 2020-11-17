#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from scipy.ndimage.morphology import grey_dilation

class MapInflator:
    """Inflates map to improve obstacle avoidance for A*."""

    pub = None

    def __init__(self):
        s
        rospy.init_node('map_inflator', anonymous=True)

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.pub = rospy.Publisher('/map_inflated', OccupancyGrid, queue_size=10)
    
    def map_callback(self,msg):
        """
        receives new map info and inflates the map
        """
        new_map = grey_dilation(msg.data, size=(3,3))
        new_msg = OccupancyGrid()
        new_msg.data = new_map
        new_msg.info = msg.info
        self.pub.publish(new_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    inflator = MapInflator()
    inflator.run()