"""A utility module

Contains useful functions to be used by the core.
"""

import numpy as np

def box_label_to_corners(chosen_obj):
    """Return corners of a 3d box

    Convert labeled 3D box to 8 corner coords.
    """
    # get 3D bounding box corners
    w = chosen_obj.box.length # for x
    h = chosen_obj.box.width # for y
    height = chosen_obj.box.height
    z = chosen_obj.box.center_z

    # corners in non-rotated
    obj_corners = np.array([[-w/2,-h/2],[w/2,-h/2],[-w/2,h/2],[w/2,h/2]])
    # rotate
    theta = chosen_obj.box.heading
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    obj_corners_after_rot = np.matmul(rot_mat, obj_corners.transpose())
    # translate
    obj_corners_after_rot_trans = obj_corners_after_rot + np.tile(np.expand_dims(np.array([chosen_obj.box.center_x,chosen_obj.box.center_y]), axis=1), (1,4))

    obj_corners_3d_up   = np.concatenate( (obj_corners_after_rot_trans, np.tile(np.expand_dims(np.array([z+height/2]),axis=1), (1,4))), axis=0)
    obj_corners_3d_down = np.concatenate( (obj_corners_after_rot_trans, np.tile(np.expand_dims(np.array([z-height/2]),axis=1), (1,4))), axis=0)
    obj_corners_3d = np.concatenate((obj_corners_3d_up, obj_corners_3d_down), axis=1)
    return obj_corners_3d

def get_global_box(frame, chosen_obj):    
    """Return box in global frame."""
    obj_corners_3d = box_label_to_corners(chosen_obj)

    ## in global frame
    pose = np.array(frame.pose.transform).reshape(4,4) # vehicle to global
    obj_corners_3d_aug = np.concatenate((obj_corners_3d,np.ones((1,8))),axis=0)
    obj_corners_3d_global = np.matmul(pose, obj_corners_3d_aug)
    obj_corners_3d_global_standard = obj_corners_3d_global[:-1,:]
    return obj_corners_3d_global_standard

def convert_camera_gc(extrinsic_mat, pose):
    """Convert camera extrinsic to 'global to camera frame'."""
    extrinsic_mat = np.linalg.inv(extrinsic_mat)# inverse to make it vehicle to camera
    ## transform the extrinsic matrix to 'global to camera'
    extrinsic_g_to_c_mat = np.matmul(extrinsic_mat, np.linalg.inv(pose)) # inv(pose): global to vehicle, extrinsic_mat: vehicle to camera
    # switch axis to make the coords look 'standard'
    extrinsic_g_to_c_mat_standard = np.matmul(np.array([ [0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1] ]), extrinsic_g_to_c_mat)
    return extrinsic_g_to_c_mat_standard

def global_box_to_camera_image_matmul(coords, extrinsic, intrinsic):
    """Return projected points from global frame to camera plane.

    Use direct matmul. The alternative is to use OpenCV.
    """
    assert len(coords.shape)==2 and coords.shape[1] == 3, 'expect coords to be Nx3'
    coords = np.concatenate((coords, np.ones((8,1))), axis=1)

    #   print(extrinsic.shape) # 4x4
    #   print(coords.shape)    # 8x4
    coords_cam_frame = np.matmul(extrinsic, coords.transpose()) # 4x8

    # remove points behind camera
    coords_keep = coords_cam_frame[2,:]>0
    coords_cam_frame = coords_cam_frame[0:3,coords_keep]
    coords_cam_frame[0,:] /= coords_cam_frame[2,:]
    coords_cam_frame[1,:] /= coords_cam_frame[2,:]
    coords_cam_frame[2,:] /= coords_cam_frame[2,:]

    cameraMatrix = np.array([[intrinsic[0],0,intrinsic[2]],
                           [0,intrinsic[1],intrinsic[3]],
                           ])

    coords_projected = np.matmul(cameraMatrix, coords_cam_frame)
    return coords_projected.transpose() # 8x2