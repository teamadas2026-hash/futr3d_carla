from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.14',
    dataroot='data/nuscenes/rgb',
    verbose=True
)

sample = nusc.sample[100]

nusc.render_pointcloud_in_image(
    sample['token'],
    pointsensor_channel='LIDAR_TOP',
    camera_channel='CAM_FRONT'
)