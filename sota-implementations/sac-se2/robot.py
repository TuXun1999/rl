import sys
import time
import cv2
from typing import Optional
import numpy as np
from scipy import ndimage
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R
from tkinter import *
import os 



import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.api import gripper_camera_param_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.robot_command import \
    RobotCommandBuilder, RobotCommandClient, \
        block_until_arm_arrives, block_for_trajectory_cmd, blocking_stand
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client import frame_helpers
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, 
                                         VISION_FRAME_NAME, 
                                         HAND_FRAME_NAME,
                                         ODOM_FRAME_NAME,
                                         GRAV_ALIGNED_BODY_FRAME_NAME,
                                         get_se2_a_tform_b,
                                         get_vision_tform_body,
                                         get_a_tform_b)

import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncPeriodicQuery
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError

from bosdyn.util import duration_str, format_metric, secs_to_hms

## RL environments
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
    ObservationNorm,
    CatTensors,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
import torch
from tensordict import TensorDict, TensorDictBase
# Hyperparameters for SPOT
VELOCITY_CMD_DURATION = 0.5  # seconds
COMMAND_INPUT_RATE = 0.1
VELOCITY_HAND_NORMALIZED = 0.5  # normalized hand velocity [0,1]
VELOCITY_ANGULAR_HAND = 1.0  # rad/sec

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_color_image': 0
}

# Hyperparameters for RL for SPOT
DEFAULT_X = 2.0
DEFAULT_ANGLE = np.pi

def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def hand_image_resolution(gripper_param_client, resolution):
    camera_mode = None
    if resolution is not None:
        if resolution == '640x480':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480
        elif resolution == '1280x720':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1280_720
        elif resolution == '1920x1080':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1920_1080
        elif resolution == '3840x2160':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_3840_2160
        elif resolution == '4096x2160':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4096_2160
        elif resolution == '4208x3120':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4208_3120

    request = gripper_camera_param_pb2.GripperCameraParamRequest(
        params=gripper_camera_param_pb2.GripperCameraParams(camera_mode=camera_mode))
    response = gripper_param_client.set_camera_params(request)


class SPOT:
    def __init__(self, options):
		# Create robot object with an image client.
        sdk = bosdyn.client.create_standard_sdk('auto_se2_task_rl')
        self.robot = sdk.create_robot(options.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.sync_with_directory()
        self.robot.time_sync.wait_for_sync()

        self.gripper_param_client = self.robot.ensure_client(\
            GripperCameraParamClient.default_service_name)
        # Optionally set the resolution of the hand camera
        if hasattr(options, 'image_sources') and 'hand_color_image' in options.image_sources:
            hand_image_resolution(self.gripper_param_client, options.resolution)
        
        if hasattr(options, 'image_service'):
            self.image_client = self.robot.ensure_client(options.image_service)
        else:
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)

        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        # Verification before the formal task
        assert self.robot.has_arm(), 'Robot requires an arm to run this example.'
        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self.robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                        'such as the estop SDK example, to configure E-Stop.'
    def lease_alive(self):
        self._lease_alive = bosdyn.client.lease.LeaseKeepAlive(\
            self.lease_client, must_acquire=True, return_at_exit=True)
        return self._lease_alive
    def lease_return(self):
        self._lease_alive.shutdown()
        self._lease_alive = None
        return self._lease_alive
    def power_on_stand(self):
        # Power on the robot
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'

        
        blocking_stand(self.command_client, timeout_sec=10)
        self.robot.logger.info('Robot standing.')
        # Command the robot to open its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)

        # Send the trajectory to the robot.
        cmd_id = self.command_client.robot_command(robot_command)

        time.sleep(4)
    def get_walking_params(self, max_linear_vel, max_rotation_vel):
        max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
        max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                            angular=max_rotation_vel)
        vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(vel_limit)
        return params
    def list_image_sources(self):
        image_sources = self.image_client.list_image_sources()
        print('Image sources:')
        for source in image_sources:
            print('\t' + source.name)
    def get_base_pose_se2(self, frame_name = VISION_FRAME_NAME):
        # The function to get the robot's base pose in SE2
        robot_state = self.state_client.get_robot_state()
        odom_T_base = frame_helpers.get_a_tform_b(\
            robot_state.kinematic_state.transforms_snapshot, frame_name, GRAV_ALIGNED_BODY_FRAME_NAME)
        return odom_T_base.get_closest_se2_transform()
    
    def send_velocit_command_se2(self, vx, vy, vtheta, exec_time = 1.5):
        # The function to send the se2 synchro velocity command to the robot
        move_cmd = RobotCommandBuilder.synchro_velocity_command(\
            v_x=vx, v_y=vy, v_rot=vtheta)

        cmd_id = self.command_client.robot_command(command=move_cmd,
                            end_time_secs=time.time() + exec_time)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=4)
    def send_pose_command_se2(self, x, y, theta, exec_time = 1.5, frame_name = VISION_FRAME_NAME):
        # The function to send the pose command to move the robot to the desired pose
        move_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(\
            goal_x=x, goal_y=y, goal_heading=theta, \
                frame_name=frame_name, \
                params=self.get_walking_params(0.6, 1)
            )
        cmd_id = self.command_client.robot_command(command=move_cmd,\
                                end_time_secs = time.time() + exec_time)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=2.0)
    def capture_images(self, options):
        # Capture and save images to disk
        pixel_format = pixel_format_string_to_enum(options.pixel_format)
        image_request = [
            build_image_request(source, pixel_format=pixel_format)
            for source in options.image_sources
        ]
        image_responses = self.image_client.get_image(image_request)
        images = []
        image_extensions = []
        for image in image_responses:
            num_bytes = 1  # Assume a default of 1 byte encodings.
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
                extension = '.png'
            else:
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    num_bytes = 3
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    num_bytes = 4
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    num_bytes = 1
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                    num_bytes = 2
                dtype = np.uint8
                extension = '.jpg'

            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                try:
                    # Attempt to reshape array into a RGB rows X cols shape.
                    img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
                except ValueError:
                    # Unable to reshape the image data, trying a regular decode.
                    img = cv2.imdecode(img, -1)
            else:
                img = cv2.imdecode(img, -1)

            if options.auto_rotate:
                img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

            # Append the image to the list
            images.append(img)
            image_extensions.append(extension)
        # # Save the image from the GetImage request to the current directory with the filename
        # # matching that of the image source.
        # image_saved_path = image.source.name
        # image_saved_path = image_saved_path.replace(
        #     '/', '')  # Remove any slashes from the filename the image is saved at locally.
        # cv2.imwrite(image_saved_path + custom_tag + extension, img)
        return image_responses, images, image_extensions
    
    def estimate_obj_distance(self, image_depth_response, bbox):
        ## Estimate the distance to the target object by estimating the depth image
        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_depth_response.shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_depth_response.shot.image.rows,
                                    image_depth_response.shot.image.cols)
        
        # Visualize the depth image
        # cv2.applyColorMap() only supports 8-bit; 
        # convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)


        # Save the image locally
        filename = os.path.join("images", "initial_search_depth.png")
        cv2.imwrite(filename, depth_color)
        # Convert the value into real-world distance & Find the average value within bbox
        dist_avg = 0
        dist_count = 0
        dist_scale = image_depth_response.source.depth_scale

        for i in range(int(bbox[1]), int(bbox[3])):
            for j in range((int(bbox[0])), int(bbox[2])):
                if cv_depth[i, j] != 0:
                    dist_avg += cv_depth[i, j] / dist_scale
                    dist_count += 1
        
        distance = dist_avg / dist_count
        return distance

    def estimate_obj_pose_hand(self, bbox, image_response, distance):
        ## Estimate the target object pose (indicated by the bounding box) in hand frame
        bbox_center = [int((bbox[1] + bbox[3])/2), int((bbox[0] + bbox[2])/2)]
        pick_x, pick_y = bbox_center
        # Obtain the camera inforamtion
        camera_info = image_response.source
        
        w = camera_info.cols
        h = camera_info.rows
        fl_x = camera_info.pinhole.intrinsics.focal_length.x
        k1= camera_info.pinhole.intrinsics.skew.x
        cx = camera_info.pinhole.intrinsics.principal_point.x
        fl_y = camera_info.pinhole.intrinsics.focal_length.y
        k2 = camera_info.pinhole.intrinsics.skew.y
        cy = camera_info.pinhole.intrinsics.principal_point.y

        pinhole_camera_proj = np.array([
            [fl_x, 0, cx, 0],
            [0, fl_y, cy, 0],
            [0, 0, 1, 0]
        ])
        pinhole_camera_proj = np.float32(pinhole_camera_proj) # Converted into float type

        # Calculate the object's pose in hand camera frame
        initial_guess = [1, 1, 10]
        def equations(vars):
            x, y, z = vars
            eq = [
                pinhole_camera_proj[0][0] * x + pinhole_camera_proj[0][1] * y + pinhole_camera_proj[0][2] * z - pick_x * z,
                pinhole_camera_proj[1][0] * x + pinhole_camera_proj[1][1] * y + pinhole_camera_proj[1][2] * z - pick_y * z,
                x * x + y * y + z * z - distance * distance
            ]
            return eq

        root = fsolve(equations, initial_guess)
        # Correct the frame conventions in hand frame & pinhole model
        # pinhole model: z-> towards object, x-> rightward, y-> downward
        # hand frame in SPOT: x-> towards object, y->rightward
        result = SE3Pose(x=root[2], y=-root[0], z=-root[1], rot=Quat(w=1, x=0, y=0, z=0))
        return result
    def correct_body(self, obj_pose_hand):
        # Find the object pose in body frame
        robot_state = self.state_client.get_robot_state()
        body_T_hand = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                    frame_helpers.HAND_FRAME_NAME)
        body_T_obj = body_T_hand * obj_pose_hand

        # Rotate the body 
        body_T_obj_se2 = body_T_obj.get_closest_se2_transform()

        # Command the robot to rotate its body
        move_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(\
            0, 0, body_T_obj_se2.angle, \
                robot_state.kinematic_state.transforms_snapshot, \
                params=self.get_walking_params(0.6, 1))
        id = self.command_client.robot_command(command=move_command, \
                                               end_time_secs=time.time() + 10)
        block_for_trajectory_cmd(self.command_client,
                                    cmd_id=id, 
                                    feedback_interval_secs=5, 
                                    timeout_sec=10,
                                    logger=None)



    def move_base_arc(self, obj_pose_hand:SE3Pose, angle):
        # Find the object pose in body frame
        robot_state = self.state_client.get_robot_state()
        body_T_hand = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                    frame_helpers.HAND_FRAME_NAME)
        body_T_obj = body_T_hand * obj_pose_hand

        # Rotate the robot base w.r.t the object pose
        body_T_obj_mat = body_T_obj.to_matrix()
        rot_mat = R.from_rotvec([0, 0, angle]).as_matrix()
        rot = np.eye(4)
        rot[0:3, 0:3] = rot_mat
        body_T_target = body_T_obj_mat @ rot @ np.linalg.inv(body_T_obj_mat)
        body_T_target = SE3Pose.from_matrix(body_T_target)
        odom_T_body = frame_helpers.get_a_tform_b(\
                    robot_state.kinematic_state.transforms_snapshot,
                    frame_helpers.VISION_FRAME_NAME, \
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)
        odom_T_target = odom_T_body * body_T_target
        # Send the command to move the robot base
        odom_T_target_se2 = odom_T_target.get_closest_se2_transform()
        # Command the robot to open its gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        move_command = RobotCommandBuilder.synchro_se2_trajectory_point_command(\
            odom_T_target_se2.x, odom_T_target_se2.y, odom_T_target_se2.angle, \
                frame_name=VISION_FRAME_NAME, \
                params=self.get_walking_params(0.6, 1),\
                build_on_command=gripper_command)
        id = self.command_client.robot_command(command=move_command, \
                                               end_time_secs=time.time() + 10)
        block_for_trajectory_cmd(self.command_client,
                                    cmd_id=id, 
                                    feedback_interval_secs=5, 
                                    timeout_sec=10,
                                    logger=None)
        print("Moving done")

    def arm_control_wasd(self):
        # Print helper messages
        print("[wasd]: Radial/Azimuthal control")
        print("[rf]: Up/Down control")
        print("[uo]: X-axis rotation control")
        print("[ik]: Y-axis rotation control")
        print("[jl]: Z-axis rotation control")
        # Use tk to read user input and adjust arm poses
        root = Tk()

        root.bind("<KeyPress>", self.arm_control)
        root.mainloop()
    def arm_control(self, event):
        # Control arm by sending commands
        if event.keysym =='w':
            self._move_out()
        elif event.keysym == 's':
            self._move_in()
        elif event.keysym == 'a':
            self._rotate_ccw()
        elif event.keysym == 'd':
            self._rotate_cw()
        elif event.keysym == 'r':
            self._move_up()
        elif event.keysym == 'f':
            self._move_down()
        elif event.keysym == 'i':
            self._rotate_plus_ry()
        elif event.keysym == 'k':
            self._rotate_minus_ry()
        elif event.keysym == 'u':
            self._rotate_plus_rx()
        elif event.keysym == 'o':
            self._rotate_minus_ry()
        elif event.keysym == 'j':
            self._rotate_plus_rz()
        elif event.keysym == 'l':
            self._rotate_minus_rz()
        elif event.keysym == 'g':
            self._arm_stow()
    def _arm_stow(self):
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = self.command_client.robot_command(stow)

        block_until_arm_arrives(self.command_client, stow_command_id, 3.0)
        
    def _move_out(self):
        self._arm_cylindrical_velocity_cmd_helper('move_out', v_r=VELOCITY_HAND_NORMALIZED)

    def _move_in(self):
        self._arm_cylindrical_velocity_cmd_helper('move_in', v_r=-VELOCITY_HAND_NORMALIZED)

    def _rotate_ccw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_ccw', v_theta=VELOCITY_HAND_NORMALIZED)

    def _rotate_cw(self):
        self._arm_cylindrical_velocity_cmd_helper('rotate_cw', v_theta=-VELOCITY_HAND_NORMALIZED)

    def _move_up(self):
        self._arm_cylindrical_velocity_cmd_helper('move_up', v_z=VELOCITY_HAND_NORMALIZED)

    def _move_down(self):
        self._arm_cylindrical_velocity_cmd_helper('move_down', v_z=-VELOCITY_HAND_NORMALIZED)

    def _rotate_plus_rx(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rx(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_ry(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_ry(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_rz(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rz(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)

    def _arm_cylindrical_velocity_cmd_helper(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """ Helper function to build a arm velocity command from unitless cylindrical coordinates.

        params:
        + desc: string description of the desired command
        + v_r: normalized velocity in R-axis to move hand towards/away from shoulder in range [-1.0,1.0]
        + v_theta: normalized velocity in theta-axis to rotate hand clockwise/counter-clockwise around the shoulder in range [-1.0,1.0]
        + v_z: normalized velocity in Z-axis to raise/lower the hand in range [-1.0,1.0]

        """
        # Build the linear velocity command specified in a cylindrical coordinate system
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self.robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_angular_velocity_cmd_helper(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """ Helper function to build a arm velocity command from angular velocities measured with respect
            to the odom frame, expressed in the hand frame.

        params:
        + desc: string description of the desired command
        + v_rx: angular velocity about X-axis in units rad/sec
        + v_ry: angular velocity about Y-axis in units rad/sec
        + v_rz: angular velocity about Z-axis in units rad/sec

        """
        # Specify a zero linear velocity of the hand. This can either be in a cylindrical or Cartesian coordinate system.
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()

        # Build the angular velocity command of the hand
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(x=v_rx, y=v_ry, z=v_rz)

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self.robot.time_sync.robot_timestamp_from_local_secs(time.time() +
                                                                           VELOCITY_CMD_DURATION))

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_velocity_cmd_helper(self, arm_velocity_command, desc=''):

        # Build synchronized robot command
        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
            arm_velocity_command)

        self._start_robot_command(desc, robot_command,
                                  end_time_secs=time.time() + VELOCITY_CMD_DURATION)
    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            self.command_client.robot_command(command=command_proto,
                                                     end_time_secs=end_time_secs)

        self._try_grpc(desc, _start_command)
    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            print(f'Failed {desc}: {err}')
            return None

## Environment for Spot in RL
# Reset the seed
def _set_seed(self, seed: Optional[int]):
    rng = torch.Generator(device=self.device)
    rng = rng.manual_seed(seed)
    self.rng = rng
# Define the parameters
def gen_params(self, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such 
    as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_velocity": 0.6,
                    "max_angular_velocity": 1.0,
                    "max_range": 1.5,
                    "max_angle": np.pi,
                    "dt": 1.0,
                },
                [],
            )
        },
        [],
        device=self.device
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


# Helper function to make a composite from tensordict
def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

class SpotRLEnvSE2(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    } # Seems to be related to rendering
    batch_locked = False # Usually set to false for "stateless" environment

    def __init__(self, robot, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        self.robot = robot

    # Helpers: _make_step and gen_params
    gen_params = gen_params
    _set_seed = _set_seed



    # Create the specification for observation, state, action
    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            x=Bounded(
                low=-torch.tensor(DEFAULT_X),
                high=torch.tensor(DEFAULT_X),
                shape=(),
                dtype=torch.float32,
            ),
            y=Bounded(
                low=-torch.tensor(DEFAULT_X),
                high=torch.tensor(DEFAULT_X),
                shape=(),
                dtype=torch.float32,
            ),
            theta=Bounded(
                low=-torch.tensor(DEFAULT_ANGLE),
                high=torch.tensor(DEFAULT_ANGLE),
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        # NOTE: Might be a bug here
        self.action_spec = Bounded(
            low=np.array(\
                [-td_params["params", "max_velocity"], \
                    -td_params["params", "max_velocity"],\
                        -td_params["params", "max_angular_velocity"]]),
            high=np.array(\
                [td_params["params", "max_velocity"], \
                    td_params["params", "max_velocity"],\
                        td_params["params", "max_angular_velocity"]]),
            shape=(3,),        
            dtype=torch.float32,
        )
        

        self.reward_spec = Unbounded(shape=(*td_params.shape, 1))
    
    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty() or tensordict.get("_reset") is not None:
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=self.batch_size)

         # Reset RL environment by commanding spot to go to a random place
        high_x = torch.tensor(DEFAULT_X, device=self.device)
        high_angle = torch.tensor(DEFAULT_ANGLE, device=self.device)
        low_x = -high_x
        low_angle = -high_angle
        obs_pose = self.robot.get_base_pose_se2()
        gx = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_x - low_x)
            + low_x
        )
        gy = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_x - low_x)
            + low_x
        )
        gt = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_angle- low_angle)
            + low_angle
        )
        # Command the SPOT to go to that place
        self.robot.send_pose_command_se2(gx, gy, gt)
        obs_pose = self.robot.get_base_pose_se2()
        
        # Extract the statistics
        x = obs_pose.position.x
        y = obs_pose.position.y
        theta = obs_pose.angle

        out = TensorDict(
            {
                "x": torch.tensor(x).to(device=self.device),
                "y": torch.tensor(y).to(device=self.device),
                "theta": torch.tensor(theta).to(device=self.device),
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out
    
    def _step(self, tensordict):
        x, y, theta = tensordict["x"], tensordict["y"], tensordict["theta"]

        dt = tensordict["params", "dt"]
        # print("===Test===")
        # print([x,y,theta])
        # Read the value
        u = tensordict["action"]
        if len(u.shape) == 1:
            vx = u[0].clamp(-tensordict["params", "max_velocity"], \
                            tensordict["params", "max_velocity"])
            vy = u[1].clamp(-tensordict["params", "max_velocity"], \
                            tensordict["params", "max_velocity"])
            vtheta = u[2].clamp(-tensordict["params", "max_angular_velocity"], \
                            tensordict["params", "max_angular_velocity"])
        else:
            vx = u[:, 0].clamp(-tensordict["params", "max_velocity"], \
                            tensordict["params", "max_velocity"])
            vy = u[:, 1].clamp(-tensordict["params", "max_velocity"], \
                            tensordict["params", "max_velocity"])
            vtheta = u[:, 2].clamp(-tensordict["params", "max_angular_velocity"], \
                            tensordict["params", "max_angular_velocity"])
        dist = (x ** 2 + y ** 2)**0.5
        # costs_location = -dist + torch.exp(-dist) + torch.exp(-10*dist)
        # costs_yaw = torch.exp(-abs(theta)) + torch.exp(-10*abs(theta))
        # costs = costs_location + costs_yaw
        costs = -dist - 0.4*abs(theta)
        assert (abs(theta) >= 0).all(), "Theta is negative, which is not allowed"
        assert (dist >= 0).all(), "Distance is negative, which is not allowed"
        # print([vx, vy, vtheta])
        px = x + vx * dt
        py = y + vy * dt
        ptheta = theta + vtheta * dt
        # Limit the motion space for the robot
        px = px.clamp(-tensordict["params", "max_range"], tensordict["params", "max_range"])
        py = py.clamp(-tensordict["params", "max_range"], tensordict["params", "max_range"])
        ptheta = ptheta.clamp(-tensordict["params", "max_angle"], tensordict["params", "max_angle"])
        # Take a step in the environment
        self.robot.send_pose_command_se2(px, py, ptheta, exec_time = float(dt))
        obs_pose = self.robot.get_base_pose_se2()
        # print(obs_pose)
        # The ODE of motion of equation
        nx = obs_pose.position.x
        ny = obs_pose.position.y
        ntheta = obs_pose.angle

        # The reward is depending on the current state
        reward = costs.view(*tensordict.shape, 1) # Expand the dim to be consistent with the shape?
        done = torch.zeros_like(reward, dtype=torch.bool)
        mask = reward > -0.25 #2.5
        done[mask] = True
        out = TensorDict(
            {
                "x": torch.tensor(nx).to(device=self.device),
                "y": torch.tensor(ny).to(device=self.device),
                "theta": torch.tensor(ntheta).to(device=self.device),
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out
    def update_robot(self, robot):
        # Update the robot to use
        self.robot = robot

def create_spot_env(robot, transform_state_dict = None, device = "cpu"):
    pass
