import importlib
import json
import threading

import requests


from classfication import *

# from rest_framework.response import Response
from flask import Flask, make_response, jsonify, Response,jsonify
from flask import request
from classfication import cls_eva
import time
import share



class Thread_watch_dog (threading.Thread):
    def __init__(self, threadID, name, counter, task_id):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.task_id = task_id

    def run(self):
        pre_time = time.time()
        while(True):
            current_time = time.time()
            if(current_time - pre_time >= 5):
                pre_time = current_time
                json_data = {'instance_id':self.task_id,'progress': share.x / share.all, 'condition':3}
                print(json_data)
                # response = requests.post("/task/sync/", json_data)

def run_all(**kwargs): #问题类型， 需要测试的属性， 模型超参数的文件地址， 数据的类型
    thread_watch_dog= Thread_watch_dog(1, "Thread-1", 1, kwargs['instance_id'])
    thread_watch_dog.setDaemon(True)
    thread_watch_dog.start()
    # # print(kwargs)
    if kwargs['task_name'] == "ImageClassification":
        print(11)

        status, metric_value, info= cls_eva.class_rightness_imagenet(data_path = kwargs['dataset_path'], data = kwargs['dataset_principal'], model_path = kwargs['model_path'], parameter_path=kwargs['parameter'],num_classes=1000, metric = kwargs['perspective_metric'])
        print(metric_value)
        if status == 1:
            json_data = {'condition': status, 'fault_info': info, 'instance_id': kwargs["instance_id"]}
            # requests.post("/task/sync/", json_data)
        else:
            json_data = {'condition': status, 'fault_info': info, 'metric_value': metric_value,
                         'instance_id': kwargs['instance_id']}
            # requests.post("/task/sync/", json_data)
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