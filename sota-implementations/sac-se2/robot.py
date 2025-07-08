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
from google.protobuf import wrappers_pb2 as wrappers
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
                                         get_a_tform_b,
                                         get_odom_tform_body)

import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, \
    world_object_pb2, synchronized_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncPeriodicQuery
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.util import duration_str, format_metric, secs_to_hms
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
import graph_nav_util

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

        # Clients
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.world_object_client = self.robot.ensure_client(WorldObjectClient.default_service_name)
        
        # Clients for GraphNav service
        # Filepath for the location to put the downloaded graph and snapshots.
        if hasattr(options, "graph_path"):
            self._download_filepath = options.graph_path
            self._upload_filepath = self._download_filepath
        else:
            self._download_filepath = "./downloaded_graph"
            self._upload_filepath = self._download_filepath

        # Crate metadata for the recording session.
        if hasattr(options, "session_name"):
            session_name = options.session_name
        else:
            session_name = "test"
        user_name = self.robot._current_user
        client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
            session_name=session_name, client_username=user_name, client_id='RecordingClient',
            client_type='Python SDK')
        # Setup the recording service client.
        self._recording_client = self.robot.ensure_client(
            GraphNavRecordingServiceClient.default_service_name)

        # Create the recording environment.
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(
                client_metadata=client_metadata))

        # Setup the graph nav service client.
        self._graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)

        self._map_processing_client = self.robot.ensure_client(
            MapProcessingServiceClient.default_service_name)

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()
        
        self.graphnav_origin = SE3Pose.from_identity()
        
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
    '''
    Find & Execute the SE2 pose of the base of the robot
    '''
    
    def get_base_pose_se2_graphnav(self, frame_name = ODOM_FRAME_NAME, seed = False):
        # The function to get the robot's base pose in SE2 using waypoints in GraphNav
        state = self._graph_nav_client.get_localization_state()
        # NOTE: Unsure if this method still suffers from the motion drift issue
        # An alternative way is to use recording_tform_body
        # odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        if seed is True:
            # If seed_tform_body is available, use it
            seed_tform_body = SE3Pose.from_proto(state.localization.seed_tform_body)
            # Find the pose in the origin
            seed_origin_tform_body = self.seed_tform_origin.inverse().mult(seed_tform_body)
            odom_tform_body = seed_origin_tform_body
        else:
            # If seed_tform_body is not available, use the KO frame
            waypoint_tform_body = SE3Pose.from_proto(state.localization.waypoint_tform_body)
            waypoint_id = state.localization.waypoint_id
            waypoint = self._get_waypoint(waypoint_id)
            odom_tform_body = (SE3Pose.from_proto(waypoint.waypoint_tform_ko).inverse()).mult(waypoint_tform_body)
        return odom_tform_body.get_closest_se2_transform()
    def get_base_pose_se2(self, frame_name = ODOM_FRAME_NAME, use_world_object_service = True,\
                        graphnav = False, seed = False):
        if graphnav is False:
            # The function to get the robot's base pose in SE2
            robot_state = self.state_client.get_robot_state()
            odom_T_base = frame_helpers.get_a_tform_b(\
                robot_state.kinematic_state.transforms_snapshot, frame_name, GRAV_ALIGNED_BODY_FRAME_NAME)
            return odom_T_base.get_closest_se2_transform()
        else:
            return self.get_base_pose_se2_graphnav(seed = seed)
    
    def send_velocity_command_se2(self, vx, vy, vtheta, exec_time = 1.0):
        # The function to send the se2 synchro velocity command to the robot
        move_cmd = RobotCommandBuilder.synchro_velocity_command(\
            v_x=vx, v_y=vy, v_rot=vtheta,\
                params=self.get_walking_params(0.6, 1))
        cmd_id = self.command_client.robot_command(command=move_cmd,\
                            end_time_secs=time.time() + exec_time)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=exec_time + 0.5)
    def send_pose_command_se2(self, x, y, theta, exec_time = 1.5, frame_name = ODOM_FRAME_NAME):
        # The function to send the pose command to move the robot to the desired pose
        move_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(\
            goal_x=x, goal_y=y, goal_heading=theta, \
                frame_name=frame_name, \
                params=self.get_walking_params(0.6, 1)
            )
        cmd_id = self.command_client.robot_command(command=move_cmd,\
                                end_time_secs = time.time() + exec_time)
        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=exec_time + 0.5)

    '''
    GraphNav services
    '''
    def get_graphnav_origin(self):
        """ Returns seed_tform_body. """
        state = self._graph_nav_client.get_localization_state()
        gn_origin_tform_body = state.localization.seed_tform_body
        return SE3Pose.from_proto(gn_origin_tform_body)
    def create_graph(self, radius = 1.0, waypoint_num = 20):
        '''
        Function to create a new graph on the robot.
        '''
        # Record the starting state of the robot.
        self.odom_tform_recording = get_odom_tform_body(self.state_client.get_robot_state().kinematic_state.transforms_snapshot)
        # Clear the map on the robot.
        self._stop_recording()
        self._clear_map()
        # Start recording a new map.
        self._start_recording()
        # Create a new waypoint at the robot's current location.
        self._create_default_waypoint()
        self.seed_tform_origin = self.get_graphnav_origin()
        # Create waypoints by a random walk
        for i in range(waypoint_num):
            # Move the robot to a random location.
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            theta = np.random.uniform(0, 2 * np.pi)
            # At the start, the motion drift of odom is not severe. 
            self.send_pose_command_se2(x, y, theta, exec_time=3.0)

            # Create a waypoint at the robot's location
            self._record_waypoint("waypoint_{}".format(i))
        
        # Stop recording the map & Download the graph
        self._stop_recording()
        self._download_full_graph()
        print("Graph is downloaded !")
        
    def _should_we_start_recording(self):
        # Before starting to record, check the state of the GraphNav system.
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            # Check that the graph has waypoints. If it does, then we need to be localized to the graph
            # before starting to record
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    # Not localized to anything in the map. The best option is to clear the graph or
                    # attempt to localize to the current map.
                    # Returning false since the GraphNav system is not in the state it should be to
                    # begin recording.
                    return False
        # If there is no graph or there exists a graph that we are localized to, then it is fine to
        # start recording, so we return True.
        return True

    def _clear_map(self):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def _start_recording(self):
        """Start recording a map."""
        should_start_recording = self._should_we_start_recording()
        if not should_start_recording:
            print("The system is not in the proper state to start recording.", \
                   "Try using the graph_nav_command_line to either clear the map or", \
                   "attempt to localize to the map.")
            return
        try:
            self.graphnav_origin = self.get_graphnav_origin()
            status = self._recording_client.start_recording(
                recording_environment=self._recording_environment)
            print("Successfully started recording a map.")
        except Exception as err:
            print("Start recording failed: " + str(err))

    def _stop_recording(self):
        """Stop or pause recording a map."""
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print("Successfully stopped recording a map.")
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                # It is possible that we are not finished recording yet due to
                # background processing. Try again every 1 second.
                if first_iter:
                    print("Cleaning up recording...")
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                print("Stop recording failed: " + str(err))
                break

    def _get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print("The recording service is on.")
        else:
            print("The recording service is off.")

    def _create_default_waypoint(self, *args):
        """Create a default waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name="default")
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            print("Successfully created a waypoint.")
        else:
            print("Could not create a waypoint.")

    def _download_full_graph(self):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        self._current_graph = graph
        if graph is None:
            print("Failed to download the graph.")
            return
        self._write_full_graph(graph)
        print("Graph downloaded with {} waypoints and {} edges".format(
            len(graph.waypoints), len(graph.edges)))
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)
    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print('Loading the graph from disk into local storage...')
        with open(self._upload_filepath + '/graph', 'rb') as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f'Loaded graph has {len(self._current_graph.waypoints)} waypoints and {self._current_graph.edges} edges'
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(f'{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}',
                      'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(f'{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}',
                      'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print('Uploading the graph and snapshots to the robot...')
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print(f'Uploaded {waypoint_snapshot.id}')
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print(f'Uploaded {edge_snapshot.id}')

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print('\n')
            print(
                'Upload complete! The robot is currently not localized to the map; please localize'
                ' the robot using commands (2) or (3) before attempting a navigation command.')
    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, '/graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception:
                # Failure in downloading waypoint snapshot. Continue to next snapshot.
                print("Failed to download waypoint snapshot: " + waypoint.snapshot_id)
                continue
            self._write_bytes(self._download_filepath + '/waypoint_snapshots',
                              '/' + waypoint.snapshot_id, waypoint_snapshot.SerializeToString())
            num_waypoint_snapshots_downloaded += 1
            print("Downloaded {} of the total {} waypoint snapshots.".format(
                num_waypoint_snapshots_downloaded, len(waypoints)))

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print("Failed to download edge snapshot: " + edge.snapshot_id)
                continue
            self._write_bytes(self._download_filepath + '/edge_snapshots', '/' + edge.snapshot_id,
                              edge_snapshot.SerializeToString())
            num_edge_snapshots_downloaded += 1
            print("Downloaded {} of the total {} edge snapshots.".format(
                num_edge_snapshots_downloaded, num_to_download))

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, 'wb+') as f:
            f.write(data)
            f.close()

    def _update_graph_waypoint_and_edge_ids(self, do_print=False):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id, do_print)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""
        self._update_graph_waypoint_and_edge_ids(do_print=True)

    def _create_new_edge(self, waypoint1, waypoint2):
        """Create new edge between existing waypoints in map."""

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        from_id = graph_nav_util.find_unique_waypoint_id(waypoint1, self._current_graph,
                                                         self._current_annotation_name_to_wp_id)
        to_id = graph_nav_util.find_unique_waypoint_id(waypoint2, self._current_graph,
                                                       self._current_annotation_name_to_wp_id)

        print("Creating edge from {} to {}.".format(from_id, to_id))

        from_wp = self._get_waypoint(from_id)
        if from_wp is None:
            return

        to_wp = self._get_waypoint(to_id)
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        # print("edge transform =", new_edge.from_tform_to)

        # Send request to add edge to map
        try:
            self._recording_client.create_edge(edge=new_edge)
        except Exception as e:
            print(e)

    def _create_loop(self, *args):
        """Create edge from last waypoint to first waypoint."""

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        if len(self._current_graph.waypoints) < 2:
            print(
                "Graph contains {} waypoints -- at least two are needed to create loop.".format(
                    len(self._current_graph.waypoints)))
            return False

        sorted_waypoints = graph_nav_util.sort_waypoints_chrono(self._current_graph)
        edge_waypoints = [sorted_waypoints[-1][0], sorted_waypoints[0][0]]

        self._create_new_edge(edge_waypoints)

    def _auto_close_loops_prompt(self, *args):
        print("""
        Options:
        (0) Close all loops.
        (1) Close only fiducial-based loops.
        (2) Close only odometry-based loops.
        (q) Back.
        """)
        try:
            inputs = input('>')
        except NameError:
            return
        req_type = str.split(inputs)[0]
        close_fiducial_loops = False
        close_odometry_loops = False
        if req_type == '0':
            close_fiducial_loops = True
            close_odometry_loops = True
        elif req_type == '1':
            close_fiducial_loops = True
        elif req_type == '2':
            close_odometry_loops = True
        elif req_type == 'q':
            return
        else:
            print("Unrecognized command. Going back.")
            return
        self._auto_close_loops(close_fiducial_loops, close_odometry_loops)

    def _auto_close_loops(self, close_fiducial_loops, close_odometry_loops, *args):
        """Automatically find and close all loops in the graph."""
        response = self._map_processing_client.process_topology(
            params=map_processing_pb2.ProcessTopologyRequest.Params(
                do_fiducial_loop_closure=wrappers.BoolValue(value=close_fiducial_loops),
                do_odometry_loop_closure=wrappers.BoolValue(value=close_odometry_loops)),
            modify_map_on_server=True)
        print("Created {} new edge(s).".format(len(response.new_subgraph.edges)))
    
    def _record_waypoint(self, name):
        self._recording_client.create_waypoint(waypoint_name=name)
        print(f'Waypoint {name} Recorded.')

    def _get_waypoint(self, id):
        """Get waypoint from graph (return None if waypoint not found)"""

        if self._current_graph is None:
            self._current_graph = self._graph_nav_client.download_graph()

        for waypoint in self._current_graph.waypoints:
            if waypoint.id == id:
                return waypoint

        print('ERROR: Waypoint {} not found in graph.'.format(id))
        return None

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()
    
    '''
    Functions to control the arm of the robot
    '''
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
    '''
    Custom Long-horizon tasks
    '''
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
                    frame_helpers.ODOM_FRAME_NAME, \
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)
        odom_T_target = odom_T_body * body_T_target
        # Send the command to move the robot base
        odom_T_target_se2 = odom_T_target.get_closest_se2_transform()
        # Command the robot to open its gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        move_command = RobotCommandBuilder.synchro_se2_trajectory_point_command(\
            odom_T_target_se2.x, odom_T_target_se2.y, odom_T_target_se2.angle, \
                frame_name=ODOM_FRAME_NAME, \
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


    

## Environment for Spot in RL



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

        # Whether to use graphnav service on SPOT to determine its own pose
        self.graphnav = (self.robot._current_graph is not None)

    # Helpers: _make_step and gen_params
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
                        "max_range": 1.0,
                        "max_angle": np.pi,
                        "dt": 1.5,
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
        obs_pose = self.robot.get_base_pose_se2(graphnav = self.graphnav)
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

# The new RL environment that generates body velocities instead
class SpotRLEnvBodyVelocitySE2(EnvBase):
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

        # Whether to use graphnav service on SPOT to determine its own pose
        self.graphnav = (self.robot._current_graph is not None)

    # Helpers: _make_step and gen_params
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
                        "max_range": 1.0,
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

        # Reset RL environment by commanding spot to follow a random body velocity
        velocity_high = tensordict["params", "max_velocity"]
        velocity_angle_high = tensordict["params", "max_angular_velocity"]
        high_x = torch.tensor(velocity_high, device=self.device)
        high_angle = torch.tensor(velocity_angle_high, device=self.device)
        low_x = -high_x
        low_angle = -high_angle

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
        dt = float(tensordict["params", "dt"].cpu().numpy())
        self.robot.send_velocity_command_se2(gx, gy, gt, exec_time = dt)
        obs_pose = self.robot.get_base_pose_se2(graphnav = self.graphnav, seed = True)
        
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
        
        # Take a step in the environment
        # Send the body velocity command
        self.robot.send_velocity_command_se2(vx, vy, vtheta, exec_time = float(dt))
        # Obtain the pose in seed-origin frame
        obs_pose = self.robot.get_base_pose_se2(graphnav = self.graphnav, seed = True)
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
