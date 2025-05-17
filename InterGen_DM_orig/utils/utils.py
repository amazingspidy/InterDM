import os
import numpy as np
import torch
# import cv2
from PIL import Image
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from common.quaternion import *

from utils.rotation_conversions import *

from utils.ig_distribution import *
face_joint_indx = [2,1,17,16]
fid_l = [7,10]
fid_r = [8,11]


def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def rigid_transform(relative, data):

    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data


class MotionNormalizer():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")

        self.motion_mean = mean
        self.motion_std = std


    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x

    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x



class MotionNormalizerTorch():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x

trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def process_motion_np(motion, feet_thre, prev_frames, n_joints):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3) 
    rotations = motion[:, n_joints*3:]

    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    '''XZ at origin'''
    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init


    positions = qrot_np(root_quat_init_for_all, positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    rot_data = rotations
    
    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data, root_quat_init, root_pose_init_xz[None]



def process_motion_np_v2(motion, feet_thre, prev_frames, n_joints):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints, :3]
    rotations = motion[:, :n_joints, 3:]

    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    '''XZ at origin'''
    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init


    positions = qrot_np(root_quat_init_for_all, positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    #rot_data = rotations
    rot_data = rotations.reshape(-1, 22*6) # TODO KIM 원복할것!!
    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1]
    print("data.shape: ", data.shape)
    data = np.concatenate([data, joint_vels], axis=-1)
    print("data.shape: ", data.shape)
    print("rot_data.shape: ", rot_data.shape)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    print("data.shape: ", data.shape)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    print("data.shape: ", data.shape)
    return data, root_quat_init, root_pose_init_xz[None]



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)





def get_velocity(transl, feet_thre, prev_frames, n_joints):
    # trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
    #                      [0.0, 0.0, 1.0],
    #                      [0.0, -1.0, 0.0]])
    # face_joint_indx = [2,1,17,16]
    # (seq_len, joints_num, 3)
    # transl: torch.Tensor (seq_len, n_joints, 3)
    B, T, N, J, D = transl.shape
    #positions = transl.view(-1, n_joints, 3)
    positions = transl  
    # # trans_matrix를 torch.Tensor로 변환하고 positions에 곱하기
    # trans_matrix = torch.tensor(trans_matrix, dtype=torch.float32)
    # positions = torch.einsum("mn, tjn->tjm", trans_matrix, positions)

    # '''Put on Floor'''
    # floor_height = positions.min(dim=0)[0].min(dim=0)[0][1]  # 가장 작은 값 구하기
    # positions[:, :, 1] -= floor_height

    # '''XZ at origin'''
    # root_pos_init = positions[prev_frames]
    # root_pose_init_xz = root_pos_init[0] * torch.tensor([1.0, 0.0, 1.0])
    # positions = positions - root_pose_init_xz

    # '''All initially face Z+'''
    # r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    # across = root_pos_init[r_hip] - root_pos_init[l_hip]
    # across = across / torch.sqrt(torch.sum(across ** 2, dim=-1, keepdim=True))

    # # forward (3,), rotate around y-axis
    # forward_init = torch.cross(torch.tensor([[0.0, 1.0, 0.0]]), across.unsqueeze(0), dim=-1)
    # forward_init = forward_init / torch.sqrt(torch.sum(forward_init ** 2, dim=-1, keepdim=True))

    # target = torch.tensor([[0.0, 0.0, 1.0]])
    # root_quat_init = qbetween_np(forward_init, target)  # qbetween_np가 torch 연산으로 되어 있어야 함
    # root_quat_init_for_all = torch.ones(positions.shape[:-1] + (4,)) * root_quat_init

    # positions = qrot_np(root_quat_init_for_all, positions)  # qrot_np도 torch 연산으로 되어 있어야 함

    '''Get Joint Velocity'''
    joint_positions = positions.reshape(B, T, N, -1)
    joint_vels = positions[:, 1:] - positions[:,:-1]
    joint_vels = joint_vels.reshape(B, T-1, N, -1)

    data = joint_positions[:,:-1]
    data = torch.cat([data, joint_vels], dim=-1)

    return joint_vels

def absolute_to_local_transforms(transforms: torch.Tensor) -> torch.Tensor:
    """
    절대 회전 행렬을 고정된 키네마틱 트리를 따라 로컬 회전 행렬로 변환합니다.

    Parameters
    ----------
    transforms : torch.tensor BxFxJx3x3
        각 관절의 절대 회전 행렬 (B: 배치 크기, F: 프레임 수, J: 관절 수)

    Returns
    -------
    local_transforms : torch.tensor BxFxJx3x3
        각 관절의 로컬 회전 행렬
    """
    # 고정된 parents 텐서 (SMPL-X 기준)
    FIXED_PARENTS = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 
                                  13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 
                                  31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 
                                  47, 21, 49, 50, 21, 52, 53], dtype=torch.long, device=transforms.device)

    B, F, J = transforms.shape[0], transforms.shape[1], transforms.shape[2]

    # 로컬 회전 텐서 초기화
    local_transforms = torch.zeros_like(transforms)  # (B, F, J, 3, 3)

    # 루트 관절: 절대 회전이 로컬 회전
    local_transforms[:, :, 0] = transforms[:, :, 0]

    # 나머지 관절에 대해 로컬 회전 계산
    for i in range(1, J):
        parent_transform = transforms[:, :, FIXED_PARENTS[i]]  # (B, F, 3, 3)
        parent_inv = parent_transform.transpose(-1, -2)       # (B, F, 3, 3)
        local_transforms[:, :, i] = torch.matmul(parent_inv, transforms[:, :, i])  # (B, F, 3, 3)

    return local_transforms


def convert_intergen_notation(output):
    B, T, N, J, _ = output.shape
    transl = output[...,:3]
    rot6d = output[...,3:] # B, F, N, J, 6
    mat_rot = rot6d2rotmat(rot6d) # B, F, N, J, 3 , 3
    local_mat_rot = absolute_to_local_transforms(mat_rot) # global -> local
    local_6d_rot = rotmat2rot6d(local_mat_rot)
    global_joint_poisition = transl.reshape(B, T, N, J*3)[:,:-1] 
    global_joint_velocity =  get_velocity(transl, 0.001, 0, n_joints=22)
    
    # print(global_joint_poisition.shape)
    # print(global_joint_velocity.shape)
    #global_joint_rotation = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).view(1,1,1,1,6).repeat(B, T-1, N, J-1, 1).reshape(B, T-1, N, -1).to(output.device)
    global_joint_rotation = local_6d_rot.reshape(B, T, N, J*6)[:,:-1,:,6:] #except root joint
    
    result = torch.cat((global_joint_poisition, global_joint_velocity, global_joint_rotation), dim=-1)
    
    return result


def convert_intergen_notation_v2(output):
    B, T, N, J, _ = output.shape
    # transl = output[...,:3]
    rot6d = output[...,3:] # B, F, N, J, 6
    mat_rot = rot6d2rotmat(rot6d) # B, F, N, J, 3, 3
    local_mat_rot = absolute_to_local_transforms(mat_rot) # global -> local
    local_6d_rot = rotmat2rot6d(local_mat_rot)
    
    output = output.cpu().numpy()

    converted_motion = []
    for i in range(B):
        motion1 = output[i,:,0]
        motion2 = output[i,:,1]

        motion1, root_quat_init1, root_pos_init1 = process_motion_np_v2(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np_v2(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)

        #motion = np.concatenate((motion1[..., :22*6], motion2[..., :22*6]), axis=1) # F-1, 264
        motion = np.stack([motion1[...,:22*6], motion2[...,:22*6]], axis = 1)
        converted_motion.append(motion)
    output = np.stack(converted_motion, axis=0)
    print("output.shape: ", output.shape)
    transl = output
    transl = torch.from_numpy(transl).to(local_6d_rot.device)
   
    local_joint_rotation = local_6d_rot.reshape(B, T, N, J*6)[:,:-1,:,6:] #프레임 빼주고, except root joint
    #local_joint_rotation = local_joint_rotation.reshape(B, T-1, -1)
    #result = torch.cat((global_joint_poisition, global_joint_velocity, global_joint_rotation), dim=-1)
    result = torch.cat((transl, local_joint_rotation), dim=-1)
    print("result.shape: ", result.shape)
    return result