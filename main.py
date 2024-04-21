import pandas as pd
import numpy as np
from my_robot_slam import MyRobotSlam
from odometer import odom_values
import my_robot_slam
from RPLidarRoboticia import rplidar
import time
from RAZ_Lidar import *
from control_motor import motor_command
LIDAR_DEVICE = "/dev/ttyUSB0"
lidar = rplidar.RPLidar(LIDAR_DEVICE,baudrate=115200)
RAZ_Lidar(lidar)
lidar.connect()
lidar.start_motor()
time.sleep(1)
print("Fin de l'initialisation du Lidar\n")
MyRobot=MyRobotSlam()

for scan in lidar.iter_scans() : 
    dl,dr=1 #dl et dr sont les distances parcourues par les deux roues  données par les capteurs de l'odométrie 
    lidar_data =scan
    odom=odom_values(dl,dr)
    command=MyRobot.control(lidar_data,odom)
    motor_command(command)
    



    
    
    
    
    