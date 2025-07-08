# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""

import argparse
import sys
import time
import numpy as np

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, \
    ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from robot import SPOT
def hello_arm(config):
    """A simple example of testing the Boston Dynamics API to detect apriltags."""

    robot = SPOT(config)
    
    with robot.lease_alive():
        robot.power_on_stand()


        # Read the fiducials
        new_graph = True
        if new_graph is True:
            robot.create_graph()
        else:
            robot._upload_graph_and_snapshots()
        attempts = 1
        # Move the robot to several different places
        # Check the estimated pose from fiducials
        for i in range(attempts):
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            theta = np.random.uniform(-np.pi, np.pi)
            robot.send_pose_command_se2(x=x, y=y, theta=theta, exec_time=4.0)
            # Read the SE2 pose from fiducials
            se2pose_fid = robot.get_base_pose_se2_graphnav()
            print("==Test==")
            print("Goal pose: ")
            print([x,y,theta])
            print("Estimated pose from graphnav: ")
            print([se2pose_fid.position.x, se2pose_fid.position.y, se2pose_fid.angle])
            print("Estimated pose from spot: ")
            se2pose_spot = robot.get_base_pose_se2()
            print([se2pose_spot.position.x, se2pose_spot.position.y, se2pose_spot.angle])
    
        # print("***Check SPOT joint angles***")
        # # Read the joint angles
        # robot_state_client = robot.state_client
        # robot_state = robot_state_client.get_robot_state()
        # joint_angles = robot_state.kinematic_state.joint_states
        # for joint in joint_angles:
        #     print(joint.name)
        #     print(joint.position)


        


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        hello_arm(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)