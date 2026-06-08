from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import numpy as np

nusc = NuScenes(version='v1.14', dataroot="/mnt/c/users/team/desktop/cap/futr3d/data/nuscenes/rgb", verbose=True)
my_sample = nusc.sample[0]
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', out_path='/mnt/c/users/team/desktop/cap/futr3d/data/nuscenes/rgb/pointcloud_rendered.png')