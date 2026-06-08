import torch

img_ckpt = torch.load('/mnt/d/teamcarla/futr3d/checkpoints/cam_res101_radar_900q.pth')
state_dict1 = img_ckpt['state_dict']

pts_ckpt = torch.load('/mnt/d/teamcarla/futr3d/checkpoints/lidar_0075_900q.pth')
state_dict2 = pts_ckpt['state_dict']
# pts_head in camera checkpoint will be overwrite by lidar checkpoint
state_dict1.update(state_dict2)

merged_state_dict = state_dict1

save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, '/mnt/d/teamcarla/futr3d/checkpoints/fusion_0075_900q.pth')
