"""HiddenFootprints Core Functions

Read from Waymo records.
Project labels in a sequence to reference frames.
"""

import numpy as np
import tensorflow as tf

from .utils import get_global_box, box_label_to_corners, global_box_to_camera_image_matmul, convert_camera_gc

def read_single_frame(frame_record, open_dataset, selected_camera='FRONT'):
    """Return a dictionary for the given frame.
    
    frame['im']: image.
    frame['extrinsic']: selected camera extrinsic matrix.
    frame['intrinsic']: selected camera intrinsic matrix.
    frame['boxes_coords']: global coordinates of 8 corners of 3d labeled boxes, size Nx8x3
    frame['boxes_id']: semantic id of 3d labeled boxes. See Waymo documentation for classes.
    """
    frame = {}
    #################
    # images
    for index, image in enumerate(frame_record.images):
        if open_dataset.CameraName.Name.Name(image.name) == selected_camera:
            im = tf.image.decode_jpeg(image.image).numpy()
            frame['im'] = im
    
    #################
    # camera extrinsic (global frame to camera frame) and intrinsic
    for camera in frame_record.context.camera_calibrations:
        if open_dataset.CameraName.Name.Name(camera.name) == selected_camera:
            extrinsic_mat = np.array(camera.extrinsic.transform).reshape(4,4)# this is camera to vehicle
            extrinsic = convert_camera_gc(extrinsic_mat, np.array(frame_record.pose.transform).reshape(4,4)) # 4x4
            intrinsic = camera.intrinsic # 9
            
            frame['extrinsic'] = extrinsic
            frame['intrinsic'] = intrinsic

    #################
    # 3D boxes in global frame
    frame['boxes_coords'] = []
    frame['boxes_id'] = []
    for chosen_obj in frame_record.laser_labels:
        # convert box to 3d cube corners in global frame
        obj_corners_3d_global_standard = get_global_box(frame_record, chosen_obj) # 3x8

        frame['boxes_coords'].append(obj_corners_3d_global_standard.transpose())
        frame['boxes_id'].append(chosen_obj.type)
    frame['boxes_coords'] = np.array(frame['boxes_coords'])
    
    return frame


def propagate(reference_frame_idx, frames):    
    """Return all boxes in the segment propagated into the reference frame, shape Nx3
    
    reference_frame_propagated_labels: [id,x,y]
    """
    
    reference_frame_camera_extrinsic = frames[reference_frame_idx]['extrinsic']
    reference_frame_camera_intrinsic = frames[reference_frame_idx]['intrinsic']
    reference_frame_propagated_labels = []
    
    for source_frame_idx in range(len(frames)): # loop through all frames
        for box_idx in range(len(frames[source_frame_idx]['boxes_id'])): # for each object in current frame
            semantic_id = frames[source_frame_idx]['boxes_id'][box_idx]
            chosen_box_coords = frames[source_frame_idx]['boxes_coords'][box_idx,:].reshape((8,3))
            # project to camera frame of 8 corners
            box_coords_projected_8_corners = global_box_to_camera_image_matmul(chosen_box_coords,
                                                                               reference_frame_camera_extrinsic,
                                                                               reference_frame_camera_intrinsic)

            if box_coords_projected_8_corners.shape[0]>4: # valid projected bottom face of 3D boxes
                footprint_x, footprint_y = box_coords_projected_8_corners[4:,:].mean(axis=0)
                reference_frame_propagated_labels.append([semantic_id, footprint_x, footprint_y])     
    return np.array(reference_frame_propagated_labels)