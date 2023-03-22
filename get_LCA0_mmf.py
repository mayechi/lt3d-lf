import json
classes = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
        'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
        'pushable_pullable', 'debris', 'traffic_cone', 'barrier']

classes_many = ['car', 'truck','adult','traffic_cone','barrier']
classes_medium = ['trailer', 'bus','construction_vehicle','bicycle','motorcycle',
                  'construction_worker', 'pushable_pullable']
classes_few = ['emergency_vehicle', 'child','police_officer','stroller','personal_mobility',
               'debris']


dis_list = ['0.5/0', '1.0/0', '2.0/0', '4.0/0']
result_dict = {}
result_ap = 0
result_json_path_list = ['results/nuscenes-metrics/metrics_summary.json']
for result_json_path in result_json_path_list:
    result_ap = 0
    result = json.load(open(result_json_path, 'r'))
    for cls in classes:
        result_dict[cls] = 0
        for dis in dis_list:
            result_dict[cls] = result_dict[cls] + result['label_aps'][cls][dis]
        result_dict[cls] = result_dict[cls] / 4.0
        result_ap += result_dict[cls]
    result_ap = result_ap / (len(classes)*1.0)
    print('result_dict:', result_dict)
    print('result_ap:', result_ap)
    
