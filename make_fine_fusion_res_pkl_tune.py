import os
import pickle
import numpy as np
import cv2
import lap
from tqdm import tqdm 
import mmcv
from mmcv import Config
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from tools.fusion_open.utils import *

font = cv2.FONT_HERSHEY_SIMPLEX

# Read cfg file
config_file = "configs/centerpoint/lt3d/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py"
cfg_lidar_3d = Config.fromfile(config_file)
cfg_lidar_3d = compat_cfg(cfg_lidar_3d)

# Some flags
write_res = 1
write_format = 'json'
# write_format = 'pkl'
draw_gt = 0
get_gt = 0
get_2d = 1
show = 0

'''********* Forward off line *********'''
classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
        'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
        'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
classes_coarse = ["vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", "vehicle", 
                "pedestrian", "pedestrian", "pedestrian", "pedestrian", "pedestrian", "pedestrian", 
                "movable", "movable", "movable", "movable"]
views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0),
                                (4, 5), (5, 6), (6, 7), (7, 4),
                                (0, 4), (1, 5), (2, 6), (3, 7)]            
color_map = {0: (255, 255, 0),
            1: (0, 255, 255)}
scale_factor = 4
score_3d_thresh = 0.1
score_2d_thresh = 0.2

'''Load val info(image, img_metas ...)'''
info_path = cfg_lidar_3d.data_root + '/nuscenes_infos_val.pkl'
info_data = pickle.load(open(info_path, 'rb'))
data = mmcv.load(info_path, file_format='pkl')
info_data = list(sorted(data['infos'], key=lambda e: e['timestamp']))

'''Load results from lidar detections'''
prediction_path = './tools/fusion_open/lidar_results/prediction_filter_by_dis.pkl'
res3d_fusion = load(prediction_path)

'''Load 2D detections from YOLOV7 or DINO'''
res2d_dir = './tools/fusion_open/camera_results/DINO/nuscenes_nuimages/'

def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid,
                           np.logical_and(points[:, 0] < width,
                                          points[:, 1] < height))
    return valid

def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]   
        
    valid = np.ones((points_camera.shape[0]),dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid

def compute_iou(rec1, rec2):
    rec1 = (rec1[1], rec1[0], rec1[3], rec1[2])
    rec2 = (rec2[1], rec2[0], rec2[3], rec2[2])        
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
    return (intersect / (sum_area - intersect)) * 1.0

def make_fine_fusion(iou_thresh, score_cal_times, lidar_no_match_times):

    all_num = 0
    not_project_num = 0
    info_index = 0
    for infos in tqdm(info_data):                   
        mask_score = res3d_fusion[info_index]['pts_bbox']['scores_3d'] > score_3d_thresh
        res3d_fusion[info_index]['pts_bbox']['boxes_3d'] = res3d_fusion[info_index]['pts_bbox']['boxes_3d'][mask_score]
        res3d_fusion[info_index]['pts_bbox']['scores_3d'] = res3d_fusion[info_index]['pts_bbox']['scores_3d'][mask_score]
        res3d_fusion[info_index]['pts_bbox']['labels_3d'] = res3d_fusion[info_index]['pts_bbox']['labels_3d'][mask_score]
        pred_boxes = res3d_fusion[info_index]['pts_bbox']['boxes_3d'].tensor.numpy()

        '''Del pre 3d Boxes'''
        if draw_gt == 1:
            pred_boxes =  np.zeros((0, 3), dtype=np.float32)
            scores = np.zeros((0, 1), dtype=np.float32)

        if pred_boxes.shape[0] == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
        else:
            corners_lidar = res3d_fusion[info_index]['pts_bbox']['boxes_3d'].corners.numpy().reshape(-1, 3)
            scores = res3d_fusion[info_index]['pts_bbox']['scores_3d']
            scores = np.array(scores, dtype=np.float32)
            names = res3d_fusion[info_index]['pts_bbox']['labels_3d']
            names = np.array(names)
            names = np.array([classes[name] for name in names])      

        pred_flags = np.ones((corners_lidar.shape[0]//8,), dtype=np.bool)
        fusion_flag = np.zeros((names.shape[0]), dtype=np.bool)
        
        if pred_boxes == []:
            continue

        projection_flag = np.zeros((pred_boxes.shape[0]), dtype=np.bool_)

        for view_index, view in enumerate(views):
            result_2d_pre = []
            img = cv2.imread(infos['cams'][view]['data_path'])
            corners_img, valid_z = lidar2img(corners_lidar, infos['cams'][view])
            valid_shape = check_point_in_img(corners_img, img.shape[0], img.shape[1])
            valid_all = np.logical_and(valid_z, valid_shape)
            valid_z = valid_z.reshape(-1, 8)
            valid_shape = valid_shape.reshape(-1, 8)
            valid_all = valid_all.reshape(-1, 8)        
            corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)

            '''Generate 3D results'''
            for aid in range(valid_all.shape[0]):
                if valid_z[aid].sum() >= 1:               
                    min_col = max(min(corners_img[aid, valid_z[aid], 0].min(), img.shape[1]), 0)
                    max_col = max(min(corners_img[aid, valid_z[aid], 0].max(), img.shape[1]), 0)
                    min_row = max(min(corners_img[aid, valid_z[aid], 1].min(), img.shape[0]), 0)
                    max_row = max(min(corners_img[aid, valid_z[aid], 1].max(), img.shape[0]), 0)
                    if (max_col - min_col) == 0 or (max_row - min_row) == 0:
                        continue                  
                    result_2d_pre.append([aid, min_row, min_col, max_row, max_col])
                    if show:
                        cv2.rectangle(img, (int(min_col), int(min_row)), (int(max_col), int(max_row)), (255, 0, 0), 2)
                        cv2.putText(img, names[aid], (int(min_col), int(min_row)), font, 1, (0, 0, 255), 1)
                        cv2.putText(img, str(scores[aid])[:4], (int(min_col), int(min_row)+30), font, 1, (255, 0, 255), 1)

            '''Generate 2D results from offline'''
            if get_2d:
                result_2d_det = []
                res_2d_file_path = res2d_dir+infos['token']+"@" +view+'.txt'
                if os.path.exists(res_2d_file_path):
                    with open(res_2d_file_path, 'r') as f:
                        objs_list = f.readlines()
                        for aid, objs in enumerate(objs_list):
                            temp_list = objs.split(' ')
                            cls_id, x1, y1, x2, y2, score = temp_list
                            score = float(score[:-1])
                            if score < score_2d_thresh:
                                continue
                            name_2d = classes[int(cls_id)]
                            result_2d_det.append([name_2d, score, float(y1), float(x1), float(y2), float(x2)])
                            if show:
                                cv2.rectangle(img, (int(float(x1)), int(float(y1))), (int(float(x2)), int(float(y2))), (0, 255, 0), 2)
                                cv2.putText(img, name_2d, (int(float(x1)), int(float(y1))-35), font, 1, (0, 0, 255), 2)
                                cv2.putText(img, str(score)[:4], (int(float(x1)), int(float(y1))-10), font, 1, (0, 0, 255), 2)  
            '''3D boxes show'''
            if show:
                for aid in range(valid_all.shape[0]):
                    score = scores[aid]
                    name = names[aid]                  
                    if valid_z[aid].sum() >= 4: 
                        min_col = max(min(corners_img[aid, valid_z[aid], 0].min(), img.shape[1]), 0)
                        max_col = max(min(corners_img[aid, valid_z[aid], 0].max(), img.shape[1]), 0)
                        min_row = max(min(corners_img[aid, valid_z[aid], 1].min(), img.shape[0]), 0)
                        max_row = max(min(corners_img[aid, valid_z[aid], 1].max(), img.shape[0]), 0) 
                        if (max_col - min_col) == 0 or (max_row - min_row) == 0:
                            continue                                          
                        cv2.putText(img, name, (int(min_col), int(min_row)-35), font, 1, (0, 0, 255), 2)
                        cv2.putText(img, str(score)[:4],  (int(min_col), int(min_row)-10), font, 1, (0, 0, 255), 2)
                        for index in draw_boxes_indexes_img_view:
                                corners_img[aid, index[0]][0] = min(max(corners_img[aid, index[0]][0], 0), img.shape[1])
                                corners_img[aid, index[0]][1] = min(max(corners_img[aid, index[0]][1], 0), img.shape[0])  
                                corners_img[aid, index[1]][0] = min(max(corners_img[aid, index[1]][0], 0), img.shape[1])  
                                corners_img[aid, index[1]][1] = min(max(corners_img[aid, index[1]][1], 0), img.shape[0])                                               
                                cv2.line(img,
                                            corners_img[aid, index[0]],
                                            corners_img[aid, index[1]],
                                            color=[255, 255, 0],
                                            thickness=scale_factor) 

            if show:
                cv2.imwrite("./results/img_show_paper_better/"+infos['token']+"&"+view+".jpg", img)
            
            '''Math and cal IOU'''
            if get_2d:
                result = []  
                if len(result_2d_pre) > 0 and len(result_2d_det) > 0:
                    score_mat = np.zeros([len(result_2d_pre), len(result_2d_det)])
                    for i in range(len(result_2d_pre)):
                        for j in range(len(result_2d_det)):
                            score = compute_iou(result_2d_pre[i][1:], result_2d_det[j][2:])
                            if score > iou_thresh:
                                score_mat[i, j] = score
                    _, x, y = lap.lapjv(1-score_mat, extend_cost=True, cost_limit=1-iou_thresh)  
                    for i, j in enumerate(y):
                        if j != -1:
                            result.append([j, i, 1.0]) 
            
            '''spatio and semantic fusion'''
            if get_2d:
                index_3d_list = []
                if len(result) > 0:
                    for aid in range(len(result)):
                        result[aid][-1] = score_mat[result[aid][0], result[aid][1]]
                        index_3d = result_2d_pre[result[aid][0]][0]
                        name_2d, score_2d = result_2d_det[result[aid][1]][0], result_2d_det[result[aid][1]][1]
                        name_3d, score_3d = names[index_3d], scores[index_3d] 
                        '''You can choose coarse or fine-frained classes matching for semantic fusion'''
                        # if classes_coarse[classes.index(name_3d)] == classes_coarse[classes.index(name_2d)]:
                        if classes.index(name_3d) == classes.index(name_2d):
                            fusion_name = name_2d 

                            p = 0.1
                            fusion_2d = score_cal_times*score_2d
                            fusion_2d_no = 1 - fusion_2d
                            fusion_3d = score_3d
                            fusion_3d_no = 1 - fusion_3d                            
                            fusion_2d_3d = fusion_2d*fusion_3d/p
                            fusion_2d_3d_no = fusion_2d_no*fusion_3d_no/(1-p)
                            fusion_score = fusion_2d_3d / ((fusion_2d_3d+fusion_2d_3d_no)*1.0)                                             
                        else:
                            fusion_score = float(score_2d)
                            fusion_name = name_2d
                        projection_flag[index_3d] = True
                        index_3d_list.append(index_3d)  

                        '''Overlapping fusion'''
                        if fusion_flag[index_3d] == True:
                            if fusion_score > res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d]:
                                '''if calibrating, you should comment out them'''
                                # res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = fusion_score
                                # res3d_fusion[info_index]['pts_bbox']['labels_3d'][index_3d] = classes.index(fusion_name)
                                pass
                        else:                           
                            res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = fusion_score
                            res3d_fusion[info_index]['pts_bbox']['labels_3d'][index_3d] = classes.index(fusion_name)
                            fusion_flag[index_3d] = True 
            
                '''For no Matched obj, reducing the lidar detection socres'''
                for temp in result_2d_pre:
                    index_3d = temp[0]
                    if index_3d not in index_3d_list:
                        score_3d = scores[index_3d]
                        name_3d = names[index_3d]
                        if fusion_flag[index_3d] == True:
                            fusion_score = res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d]
                            fusion_name = classes[res3d_fusion[info_index]['pts_bbox']['labels_3d'][index_3d]]
                        else:
                            fusion_score = score_3d
                            fusion_name = name_3d
                            if fusion_name == 'car' or fusion_name == 'adult':
                                res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = float(score_3d * lidar_no_match_times)
                            else:
                                res3d_fusion[info_index]['pts_bbox']['scores_3d'][index_3d] = float(score_3d * lidar_no_match_times)
        
        all_num = all_num + pred_boxes.shape[0]
        if int(projection_flag.sum()) < pred_boxes.shape[0]:
            not_project_num = not_project_num + (pred_boxes.shape[0] - projection_flag.sum())
        info_index += 1
    not_project_scale = not_project_num * 1.0 / all_num    

    print('not_project_num:', not_project_num)
    print('all_num:', all_num)
    print('not_project_scale:', not_project_scale)

    if write_res:
        if write_format == 'pkl':
            res3d_fusion_path = "./results/prediction_fine_fusion.pkl"
            save(res3d_fusion, res3d_fusion_path)
        else:
            '''Trans format to json results'''
            from mmcv import Config
            try:
                from mmdet.utils import compat_cfg
            except ImportError:
                from mmdet3d.utils import compat_cfg
            from mmdet3d.datasets import build_dataset
            config_file = "configs/centerpoint/lt3d/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_nus.py"
            out_path = "./results"
            cfg_lidar_3d = Config.fromfile(config_file)
            cfg_lidar_3d = compat_cfg(cfg_lidar_3d)
            dataset = build_dataset(cfg_lidar_3d.data.test)
            result_files, tmp_dir = dataset.format_results(res3d_fusion, out_path, None)



