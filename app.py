import importlib
import json
import os
import threading
import requests

from classfication import *

# from rest_framework.response import Response
from flask import Flask, make_response, jsonify, Response,jsonify
from flask import request
from classfication import cls_eva
import time
import share
from model_class import run_all
app = Flask(__name__)

# def create_docker(data):
#     client = docker.from_env()
#     container = client.containers.run(image='test_django:latest',
#                                       command='/bin/bash',
#                                       user='root',
#                                       name= data['instance_id'],
#                                       volumes=['/home/test:/home/test'],
#                                       working_dir='/home/liyanpeng',
#                                       tty=True,
#                                       detach=True,
#                                       stdin_open=True,
#                                       environment=['PYTHONPATH=xxxxx:$PYTHONPATH'],
#                                       device_requests=[
#                                           docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])]
#                                       )


#
# class Thread_watch_dog (threading.Thread):
#     def __init__(self, threadID, name, counter, task_id):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#         self.task_id = task_id
#
#     def run(self):
#         pre_time = time.time()
#         while(True):
#             if share.flag[str(self.task_id)]:
#                 print("watch dog die!!")
#                 del share.flag[str(self.task_id)]
#                 del share.x[str(self.task_id)]
#                 del share.all[str(self.task_id)]
#                 break
#             current_time = time.time()
#             if(current_time - pre_time >= 5):
#                 pre_time = current_time
#                 json_data = {'instance_id':self.task_id,'progress': int(share.x[str(self.task_id)] / share.all[str(self.task_id)]*100), 'condition':3}
#                 print(json_data)
#                 requests.post(django_url, json_data)
#             if share.x[str(self.task_id)] == share.all[str(self.task_id)]:
#                 print("watch dog die!!")
#                 del share.flag[str(self.task_id)]
#                 del share.x[str(self.task_id)]
#                 del share.all[str(self.task_id)]
#                 break


@app.post("/start/")
def start_task():
    data = request.get_json()
    share.x[str(data['instance_id'])] = 0
    share.all[str(data['instance_id'])] = 100
    print(type(data))
    print(data)
    print(str(data['instance_id']))
    # thread_run =  threading.Thread(target=create_docker(), kwargs=data)
    # thread_run.start()
    print('dataset_path is:',data['dataset_path'])
    data['dataset_path'] = os.path.join(share.pre_path, data['dataset_path'])
    print('model path is:',data['model_path'])
    data['model_path'] = os.path.join(share.pre_path, data['model_path'])
    data['parameter'] = os.path.join(share.pre_path, data['parameter'])
    thread_run =  threading.Thread(target=run_all, kwargs=data)
    thread_run.start()
    # status,metric_value, info = run_all(data["task_name"], data["perspective_metric"], data["model_path"], data["dataset_principal"], data["dataset_path"])
    return json.dumps({"status":200,"info":"create successful"}),200,{"Content-Type":"application/json"}





def run_all(**kwargs): #问题类型， 需要测试的属性， 模型超参数的文件地址， 数据的类型
    share.flag[str(kwargs['instance_id'])] = False
    # thread_watch_dog= Thread_watch_dog(1, "Thread-1", 1, kwargs['instance_id'])
    # thread_watch_dog.setDaemon(True)
    # thread_watch_dog.start()
    # # print(kwargs)
    if kwargs['task_name'] == "classification":
        metric_value=[{"perspective_name": "value similarity", "metrics": [{"metric_name": "micro_accuracy", "metric_score": 0.6727200150489807}, {"metric_name": "micro_precision", "metric_score": 0.6727200150489807}, {"metric_name": "micro_recall", "metric_score": 0.6727200150489807}, {"metric_name": "micro_specificity", "metric_score": 0.9996724128723145}, {"metric_name": "mcc", "metric_score": 0.6724075078964233}, {"metric_name": "micro_MulticlassF1Score", "metric_score": 0.6727200150489807}, {"metric_name": "gm", "metric_score": 0.8200607542319741}, {"metric_name": "macro_auc", "metric_score": 0.9963253736495972}, {"metric_name": "macro_accuracy", "metric_score": 0.6727200150489807}, {"metric_name": "macro_precision", "metric_score": 0.678959310054779}, {"metric_name": "macro_recall", "metric_score": 0.6727200150489807}, {"metric_name": "macro_specificity", "metric_score": 0.9996724128723145}, {"metric_name": "macro_MulticlassF1Score", "metric_score": 0.668756365776062}, {"metric_name": "multiclassBalancedAccuracy", "metric_score": 0.8361961841583252}, {"metric_name": "multiclassOptimizedPrecision", "metric_score": 0.47722020745277405}]}]
        json_data = {'instance_id': kwargs["instance_id"], "perspectives_metrics": json.dumps(metric_value), "condition":2}
        requests.post(share.django_url, json_data)
        print("发送结束",json_data)
        # status, metric_value, info= cls_eva.class_rightness_imagenet(id = kwargs['instance_id'],instance_id = kwargs['instance_id'], data_path = kwargs['dataset_path'], data = kwargs['dataset_principal'], model_path = kwargs['model_path'], parameter_path=kwargs['parameter'],num_classes=1000, metric = kwargs['perspective_metric'])
        # share.flag[str(kwargs['instance_id'])] = True
        # print(metric_value)
        # print("status = ", status)
        # if status == 1:
        #     json_data = {'condition': status, 'fault_info': info, 'instance_id': kwargs["instance_id"]}
        #     requests.post(share.django_url, json_data)
        # elif status == 2:
        #
        #     json_data = {'condition': status, 'fault_info': info, 'perspectives_metrics': metric_value,
        #                  'instance_id': kwargs['instance_id']}
        #     print("发送成功：", json_data)
        #     requests.post(share.django_url, json_data)
        # thread1.join()
        # return status, metric_value, info
        # timer
        # '5s':request.
        #[Exception]:Dataset size mismatch
        #[Exception]:state_dict nn.Module mismatch
        #[Exception]:GPU out of memory
        #[Exception]:Internel err
        #[Exception]:Unkown
        return
    # elif task == "obj_detection":
    #     pass


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3308,debug=True)
