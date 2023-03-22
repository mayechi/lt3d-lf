eval_cls_list = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
                 'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
                 'pushable_pullable', 'debris', 'traffic_cone', 'barrier']
score_cal_times_method_dict = {}
score_cal_times_dict_bay = {}

'''Different 3D or 2D detections have different calibration factors'''
# DINO Centerpoint cal super_class
for cls in eval_cls_list:
    if cls == 'car':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1
    elif cls == 'truck':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'trailer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.5
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bus':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.8
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bicycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1         
    elif cls == 'motorcycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'emergency_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.9
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'adult':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.2         
    elif cls == 'child':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 4.1
        score_cal_times_dict_bay[cls]['p'] = 0.1           
        # score_cal_times_dict_bay[cls] = 4.1
    elif cls == 'police_officer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.8
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_worker':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.6
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'stroller':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.5
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'personal_mobility':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.9
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'pushable_pullable':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.4
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'debris':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.6
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'traffic_cone':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'barrier':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        

# DINO Centerpoint cal fine_grained_class
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
        # score_cal_times_dict_bay[cls] = 4.1
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

# DINO Transfusion cal super_class
for cls in eval_cls_list:
    if cls == 'car':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.9
        score_cal_times_dict_bay[cls]['p'] = 0.1
    elif cls == 'truck':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'trailer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.3
        score_cal_times_dict_bay[cls]['p'] = 0.1 
    elif cls == 'bus':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.7
    elif cls == 'construction_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bicycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1         
    elif cls == 'motorcycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'emergency_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 2.0
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'adult':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'child':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 4.6
        score_cal_times_dict_bay[cls]['p'] = 0.1           
    elif cls == 'police_officer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.7
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_worker':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.3
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'stroller':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.7
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'personal_mobility':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.9
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'pushable_pullable':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.4
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'debris':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.9
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'traffic_cone':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'barrier':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        

# YOLOV7 Transfusion cal super_class
for cls in eval_cls_list:
    if cls == 'car':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1
    elif cls == 'truck':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.9
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'trailer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bus':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'bicycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.1
        score_cal_times_dict_bay[cls]['p'] = 0.1         
    elif cls == 'motorcycle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.0
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'emergency_vehicle':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 2.0
        score_cal_times_dict_bay[cls]['p'] = 0.1          
    elif cls == 'adult':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.2         
    elif cls == 'child':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 4.6
        score_cal_times_dict_bay[cls]['p'] = 0.1           
    elif cls == 'police_officer':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.8
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'construction_worker':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.3
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'stroller':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.6
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'personal_mobility':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.9
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'pushable_pullable':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.3
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'debris':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.7
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'traffic_cone':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 1.2
        score_cal_times_dict_bay[cls]['p'] = 0.1        
    elif cls == 'barrier':
        score_cal_times_dict_bay[cls] = {}
        score_cal_times_dict_bay[cls]['c'] = 0.1
        score_cal_times_dict_bay[cls]['p'] = 0.1        