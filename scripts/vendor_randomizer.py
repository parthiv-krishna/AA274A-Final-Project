#!/usr/bin/env python

import random
import rospkg
import rospy

# files for i/o of randomization (offset from package root)
TEMPLATE = "/models/foods/materials/scripts/template.txt"
OUTPUT   = "/models/foods/materials/scripts/randomized.material"

# options for image filepaths
image_paths = ["broccoli.png", "banana.png", "apple.png"]

def main():
    # get absolute path of package
    rospack = rospkg.RosPack()
    package_dir = rospack.get_path("final_project")
    
    # randomize vendor locations
    random.shuffle(image_paths)
    print("Randomized vendor locations: " + str(image_paths))
    
    line_num = 1
    with open(package_dir + TEMPLATE, "r") as template:
        with open(package_dir + OUTPUT, "w") as output:
            for line in template:
                if line_num % 15 == 10: # lines of interest are 10, 25, 40
                    image_file = image_paths[line_num // 15]
                    output.write("                texture " + image_file  + "\n")
                else:
                    output.write(line)  # otherwise just write the existing line
                line_num += 1

if __name__ == "__main__":
    main()
