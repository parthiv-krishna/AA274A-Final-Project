#!/usr/bin/env python

import rospy
import rosnode
from geometry_msgs.msg import PointStamped
from final_project.msg import Order

class OrderPlacer:
    """Publishes food delivery requests to the PavoneCart food delivery service. """

    request_location = None
    got_location = False

    def __init__(self):
        #initialize node
        rospy.init_node('order_placer', anonymous=True)
        rospy.Subscriber('/clicked_point', PointStamped, self.point_callback)
        #create publisher
        self.request_publisher = rospy.Publisher('/order', Order, queue_size=10)

    def publish_request(self, foods):
        order = Order()
        order.foods = foods
        order.location = self.request_location
        got_location = False
        self.request_publisher.publish(order)
        print("Published order for " + str(order.foods) + " to location \n" + str(order.location))
            
    def point_callback(self, msg):
        self.request_location = msg.point
        #print("Received point:\n" + str(self.request_location))
        self.got_location = True
    
    def loop(self):
        """The main loop of the script. The script will ask for food items to add to the 
        delivery_request string until an empty answer ("") is given, at which point it will 
        publish the string. The current request will be published several few times after which the user 
        will be prompted to create a new request."""
        if "/explore_control" in rosnode.get_node_names():
            print("Robot is currently exploring. Please wait for exploration to finish")
            return
        

        foods = []
        new_item = raw_input("Add an item to your order: ")
        while new_item != "":
            valid_foods = ["broccoli", "apple", "banana"]
            if new_item in valid_foods:
                if new_item not in foods:
                    foods.append(new_item)
                else:
                    print(new_item + " is already in the order")
            else:
                print(new_item + " is not a valid food . Please enter one of the following: " + str(valid_foods))
            new_item = raw_input("Add an item to your order: ")
       
        if len(foods) > 0:
            print("Please use the Publish Point option in rviz to set a delivery location")
            rate = rospy.Rate(0.5)
            self.got_location = False
            while not self.got_location:
                print("Waiting for point to be published on /clicked_point...")
                rate.sleep()
                
             
            self.publish_request(foods)

        print("\n Publish another request?")

    def run(self):
        print "Create a delivery request:"
        print "You'll be prompted to enter food items one at a time. Once your order list is " \
            "complete, press enter to send your order."
        rate = rospy.Rate(0.2) # 0.2 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

if __name__ == '__main__':
    may_i_take_your_order = OrderPlacer()
    may_i_take_your_order.run()
