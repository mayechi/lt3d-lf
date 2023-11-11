from tools.fusion.make_fine_fusion_res_pkl_tune import make_fine_fusion
from tools.fusion.eval_fusion_res_tune import eval_fusion
from tools.fusion.get_LCA0 import LCA0

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

score_cal_times_dict_max = {}
for cls in eval_cls_list:
    if cls == 'car':
        score_cal_times_dict_max[cls] = 0.7
    elif cls == 'truck':
        score_cal_times_dict_max[cls] = 1.5
    elif cls == 'trailer':
        score_cal_times_dict_max[cls] = 0.9
    elif cls == 'bus':
        score_cal_times_dict_max[cls] = 0.8
    elif cls == 'construction_vehicle':
        score_cal_times_dict_max[cls] = 1.2
    elif cls == 'bicycle':
        score_cal_times_dict_max[cls] = 1.1
    elif cls == 'motorcycle':
        score_cal_times_dict_max[cls] = 1.0
    elif cls == 'emergency_vehicle':
        score_cal_times_dict_max[cls] = 1.7
    elif cls == 'adult':
        score_cal_times_dict_max[cls] = 1.1
    elif cls == 'child':
        score_cal_times_dict_max[cls] = 4.1
    elif cls == 'police_officer':
        score_cal_times_dict_max[cls] = 2.0
    elif cls == 'construction_worker':
        score_cal_times_dict_max[cls] = 1.5
    elif cls == 'stroller':
        score_cal_times_dict_max[cls] = 1.8
    elif cls == 'personal_mobility':
        score_cal_times_dict_max[cls] = 1.8
    elif cls == 'pushable_pullable':
        score_cal_times_dict_max[cls] = 1.4
    elif cls == 'debris':
        score_cal_times_dict_max[cls] = 0.1
    elif cls == 'traffic_cone':
        score_cal_times_dict_max[cls] = 1.1
    elif cls == 'barrier':
        score_cal_times_dict_max[cls] = 1.4
score_cal_times_method_dict['max'] = score_cal_times_dict_max

# Tune the cal factor
res_dict = {}
for eval_cls in eval_cls_list:
    res_dict[eval_cls] = {}
for eval_cls in eval_cls_list:
    result_ap_list = []
    score_cal_times_list = []
    score_cal_times_ori = score_cal_times_method_dict['bay'][eval_cls]['c']
    score_cal_times = score_cal_times_method_dict['bay'][eval_cls]['c']
    # score_cal_times_ori = score_cal_times_method_dict['max'][eval_cls]
    # score_cal_times = score_cal_times_method_dict['max'][eval_cls]
    while True:
        print('#######################'+eval_cls+'#######################')
        print('########################'+str(score_cal_times)+'#######################')
        score_cal_times_list.append(score_cal_times) 
        make_fine_fusion(0.5, score_cal_times, 0.3)
        eval_fusion(eval_cls=eval_cls)
        result_ap_new = LCA0(eval_cls=eval_cls)        
        if len(result_ap_list) == 0:
            result_ap_list.append(result_ap_new)
            result_ap_ori = result_ap_new
            score_cal_times += 0.1
        elif len(result_ap_list) == 1:
            result_ap_list.append(result_ap_new)
            if result_ap_new >= result_ap_ori:
                score_cal_times += + 0.1
            else:
                score_cal_times = score_cal_times_ori - 0.1
        else:
            result_ap_max = max(result_ap_list)
            result_ap_list.append(result_ap_new)
            if result_ap_new > result_ap_max:
                if score_cal_times > score_cal_times_ori:
                    score_cal_times += 0.1
                else:
                    score_cal_times -= 0.1
            else:
                res_dict[eval_cls]['cal'] = score_cal_times_list[result_ap_list.index(max(result_ap_list))]
                res_dict[eval_cls]['score'] = max(result_ap_list)
                break

    print(res_dict)
