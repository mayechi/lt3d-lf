from tools.fusion_open.make_fine_fusion_res_done import make_fine_fusion
from tools.fusion_open.eval_fusion_res import eval_fusion
from tools.fusion_open.get_LCA0 import LCA0

eval_cls_list = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                 'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                 'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
score_cal_times_method_dict = {}

# DINO Centerpoint cal fine_grained_class for bayes
score_cal_times_dict_bay = {}
for cls in eval_cls_list:
    if cls == 'car':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.6
        score_cal_times_dict_bay[cls]['p'] = 0.1
    elif cls == 'truck':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'trailer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.6
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bus':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.5
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bicycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.8
        score_cal_times_dict_bay[cls]['p'] = 0.1         
    elif cls == 'motorcycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.7
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'emergency_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.9
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'adult':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.3
        score_cal_times_dict_bay[cls]['p'] = 0.1         
    elif cls == 'child':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 4.1
        score_cal_times_dict_bay[cls]['p'] = 0.1           
    elif cls == 'police_officer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.6
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_worker':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.4
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'stroller':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.4
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'personal_mobility':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.9
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'pushable_pullable':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'debris':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.5
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'traffic_cone':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'barrier':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1  
score_cal_times_method_dict['bay'] = score_cal_times_dict_bay

make_fine_fusion(0.5, score_cal_times_method_dict, 0.3)
eval_fusion(eval_cls='all')
LCA0(eval_cls='all')
