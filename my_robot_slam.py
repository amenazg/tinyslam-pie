"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
from time import sleep
from control import findWall,potential_field_control,reactive_obst_avoid

from TinySlam import TinySlam
    


# Definition of our robot controller
class MyRobotSlam():
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self):
        # Passing parameter to parent class

        # step counter to deal with init and display
        self.counter = 0

        # objectif
        self.objectifList = [[-250,-400,1], [-400,-500,1], [-500,-370,1], [-210,-270,1], [-300,-240,1], [-500,-300,1], [-500,-120,1]]
        self.i = 0

        self.objectif = np.array([0,0,0])

        # Init SLAM object
        self._size_area = (1200, 1200)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        self.lastScore = 0

        self.flagWallFound = False

        self.trajectory = []

        self.trajPoint = [0,0,0]
        self.traj_i = 0

    def control (self,lidar,odometer_values,index):
        """
        Main control function executed at each time step
        """
        exploratory_iter = 4700

        # initialization
        if index < 10:
            self.tiny_slam.update_map(lidar(),odometer_values())
            score = self.tiny_slam.localise(lidar(),odometer_values())
            self.last_score = score

        # general case
        else:
            score = self.tiny_slam.localise(lidar(),odometer_values())
            if score > 50:
                self.tiny_slam.update_map(lidar(),TinySlam.get_corrected_pose(odometer_values()))
                self.last_score = score
        self.tiny_slam.display2(TinySlam.get_corrected_pose(odometer_values()))


        # initial map
        if self.counter < 100:
            command = {"forward": 0, "rotation": 0}

        # wallfollow to create map
        elif self.counter < exploratory_iter:
            if (self.flagWallFound == False):
                command, self.flagWallFound = findWall(self.lidar())
                print("finding wall")
            else:
                command = reactive_obst_avoid(self.lidar())

        # follow path from A*
        else:
            # as the resolution of the trajectory is o high, sample it
            jump = 10

            pose = self.tiny_slam.get_corrected_pose(self.odometer_values())[:2]
            goal = [0,0]
            goal[0],goal[1] = self.tiny_slam._conv_map_to_world(self.trajPoint[0],self.trajPoint[1])
            dist = np.sqrt((goal[0]-pose[0])**2+(goal[1]-pose[1])**2)

            # if goal was atteined, get next one
            if(dist < 10):
                self.traj_i += jump
                if(self.traj_i > len(self.trajectory)):
                    print('We are done, press any key to end')
                    foo = input()
                    exit
                #update next goal
                self.trajPoint = self.trajectory[self.traj_i]

            command = potential_field_control(self.lidar(), np.array(self.tiny_slam.get_corrected_pose(self.odometer_values())),goal)

        # calculates A* one time
        if self.counter == exploratory_iter:
            self.trajectory = self.tiny_slam.plan(self.tiny_slam.get_corrected_pose(self.odometer_values()), [0,0])
            self.trajPoint = self.trajectory[0]

        # affichage
        if self.counter % 2 == 0:
            # without trajectory
            if self.counter <= exploratory_iter:
                self.tiny_slam.display2(self.tiny_slam.get_corrected_pose(self.odometer_values()))
            # with trajectory
            else:
                self.tiny_slam.display2(self.tiny_slam.get_corrected_pose(self.odometer_values()), self.trajectory)
                
        # flag_next_point = False
        # command, flag_next_point = potential_field_control(self.lidar(),self.odometer_values(),self.objectif)
        # if flag_next_point:
        #     if(self.i < len(self.objectifList)):
        #         self.objectif = self.objectifList[self.i]
        #         self.i += 1
        #     else:
        #         print("Done!")

        
        print(self.counter)
        return command



