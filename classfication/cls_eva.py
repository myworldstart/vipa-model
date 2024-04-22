import importlib
import sys

import share
import requests
sys.path.append("/home/xjb/code/vipa-model/")
sys.path.append("/home/xjb/code/vipa-model/cocoLRPapi-master/PythonAPI/pycocotools")
sys.path.append("/home/xjb/code/vipa-model/cocoLRPapi-master/PythonAPI")
import math
import models
import loaders
import torch
from pycm import *
from torchmetrics import Precision, Recall, Specificity
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from metrics import ildv
from sklearn import metrics
from torchmetrics.classification import MulticlassSpecificity, MulticlassAccuracy, MulticlassMatthewsCorrCoef, \
    MulticlassAUROC
import cv2
import imutils
from sklearn.metrics import matthews_corrcoef
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
# batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# sys.path.append("")
# train_dataset = datasets.MNIST(root='D:\\project\\pyproject\\djangoProject\\pythonProject\\dataset\\mnist\\', train=True, download=False, transform=transform)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# test_dataset = datasets.MNIST(root='D:\\project\\pyproject\\djangoProject\\pythonProject\\dataset\\mnist\\', train=False, download=False, transform=transform)
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# model = Net()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#
# model.load_state_dict(torch.load('C:\\Users\\25086\\Desktop\\test\\test.txt'))

num = torch.zeros(2000, 4)#tp fn fp tn
y_true = []
y_pred = []
#
# def Macro_Mcc(num, total, Max):
#     sum = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum = sum + (TP * TN - FN * FP) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
#     return sum / (Max + 1)
#
# def Macro_Gm(num, total, Max):
#     sum = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum = sum + math.sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
#     return sum / (Max + 1)
#
# def NPV(num, total, Max):
#     sum = 0
#     sum1 = 0
#     sum2 = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum = sum + TN / (TN + FN)
#         sum1 += TN
#         sum2 += FN
#     return sum / (Max + 1), sum1 / (sum2 + sum1)
#
# def FPR(num, total , Max):
#     sum = 0
#     sum1 = 0
#     sum2 = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum1 += FP
#         sum2 += TN
#         sum = sum + (FP / (FP + TN))
#     return sum1 / (sum1 + sum2), sum / (Max + 1)
#
# def FNR(num, total, Max):
#     sum = 0
#     sum1 = 0
#     sum2 = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum1 += FN
#         sum2 += TP
#         sum = sum + (FN / (FN + TP))
#     return sum1 / (sum1 + sum2), sum / (Max + 1)
#
# def FOR(num, total, Max):
#     sum = 0
#     sum1 = 0
#     sum2 = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum1 += FN
#         sum2 += TN
#         sum = sum + (FN / (FN + TN))
#     return sum1 / (sum1 + sum2), sum / (Max + 1)
#
# def FDR(num, total, Max):
#     sum = 0
#     sum1 = 0
#     sum2 = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sum1 += FP
#         sum2 += TP
#         sum = sum + (FP / (FP + TP))
#     return sum1 / (sum1 + sum2), sum / (Max + 1)
#
# def OP(num, total, Max, acc):
#     sum = 0
#     sum1 = 0
#     sum2 = 0
#     for i in range(Max + 1):
#         num[i][3] = total - num[i][0] - num[i][1] - num[i][2]
#         TP = num[i][0]
#         FN = num[i][1]
#         FP = num[i][2]
#         TN = num[i][3]
#         sen = TP / (TP + FN)
#         spe = TN / (TN + FP)
#         sum = sum + acc - (abs(sen - spe)) / (sen + spe)
#     return
# batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# test_dataset = datasets.MNIST(root='/home/xjb/docker-test/mnist', train=False, download=False, transform=transform)
# testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

def class_rightness_imagenet(id, instance_id, data_path, data, model_path, parameter_path, num_classes, metric):
    print('start')
    print(data_path)
    print(model_path)
    print(parameter_path)
    metric_value = []
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}!")
    parameters_size1 = []
    parameters_size2 = []
    try:
        spec = importlib.util.spec_from_file_location('get_model', model_path)
        ModelLib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ModelLib)
        model = ModelLib.get_model()
        print(f"Successfully get model {type(model)}!")
        # model = models.load_model(model_name="ResNet18", num_classes=num_classes)
        # for name,param  in model.named_parameters():
        #     parameters_size1.append(param.size())
        # print(parameters_size1[0])
        #dataloader  -> ndarray<img> cv2.resize((img_size,img_size))
        #random_crop(256,256)
        print(f"prepared to load model parameter from {parameter_path}!")
        state_dict = torch.load(parameter_path)
        model.load_state_dict(state_dict)
        print(f"Successfully load model parameter from {parameter_path}!")
    except RuntimeError as e:
        print(e)
        share.flag[str(instance_id)] = True
        return 1, metric_value, "state_dict nn.Module mismatch"
    except FileNotFoundError as e:
        print(e)
        share.flag[str(instance_id)] = True
        return 1, metric_value, "File not find"
    except Exception as e:
        print(e)
        share.flag[str(instance_id)] = True
        return 1, metric_value, "unknown"
    model.to(device)
    model.eval()
    # model.load_state_dict(torch.load(parameter_path))
    total = 0
    outputss = []
    labelss = []
    try:
        print("test_loader")
        test_loader = loaders.load_data(data_name=data, data_dir=data_path, data_type='test')
        # share.all[str(instance_id)] = len(test_loader)
        # print(share.all[str(instance_id)])
        all = len(test_loader)
        print("test_loader end")
    except FileNotFoundError as e:
        print(e)
        share.flag[str(instance_id)] = True
        return 1, metric_value, "File not find"
    except Exception as e:
        print(e)
        share.flag[str(instance_id)] = True
        return  1, metric_value, "Unkown"
    pre = 0;
    try:
        for i, samples in enumerate(tqdm(test_loader)):
            if i - pre >= 10:
                json_data = {'instance_id': id,'progress': int(pre / all * 100),'condition': 3}
                print("发送：",json_data)
                requests.post(share.django_url, json_data)
                pre = i
            print(i)
            inputs, labels = samples
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            outputss.append(outputs)
            labelss.append(labels)
    except RuntimeError as e:
        print(e)
        share.flag[str(instance_id)] = True
        return  1, metric_value, "Internel err"
    except Exception as e:
        print(e)
        share.flag[str(instance_id)] = True
        return  1, metric_value, "Unkown"

    # device=torch.device('cpu')
    output = torch.cat(outputss,dim=0)
    output.to(device)
    label = torch.cat(labelss, dim=0)
    label.to(device)
    json_data = {'instance_id': id, 'progress': 100, 'condition': 3}
    requests.post(share.django_url, json_data)
    # share.x[str(instance_id)] = share.all[str(instance_id)]
    # metric.to(device)
    # total = len(label)
    # for i in range(len(label)):
    #     y_true = label[i]
    #     if(output[i] == y_true):
    #         num[label][0] = num[label][0] + 1
    #     else:
    #         num[label][1] += 1
    #         num[output[i]][2] += 1
#-----------------------------------------------------------------------------------------------
    for i in range(len(metric)):
        for j in range(len(metric[i]['metrics'])):
            if metric[i]['metrics'][j]['metric_name'] == 'micro_accuracy':
                micro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
                metric[i]['metrics'][j]['metric_score'] = micro_accuracy(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'macro_accuracy':
                macro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
                metric[i]['metrics'][j]['metric_score'] = macro_accuracy(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'micro_precision':
                micro_precision = Precision(task="multiclass", average='micro', num_classes=num_classes).to(device)
                metric[i]['metrics'][j]['metric_score'] = micro_precision(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'macro_precision':
                macro_precision = Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
                metric[i]['metrics'][j]['metric_score'] = macro_precision(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'micro_recall':
                micro_recall = Recall(task='multiclass', average='micro', num_classes=num_classes).to(device)
                metric[i]['metrics'][j]['metric_score'] = micro_recall(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'macro_recall':
                macro_recall = Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
                metric[i]['metrics'][j]['metric_score']  = macro_recall(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'micro_specificity':
                micro_specificity = Specificity(task='multiclass', average='micro', num_classes=num_classes).to(device)
                micro_specificity.update(output, label)
                metric[i]['metrics'][j]['metric_score'] = micro_specificity(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'macro_specificity':
                macro_specificity = Specificity(task='multiclass', average='macro', num_classes=num_classes).to(device)
                macro_specificity.update(output, label)
                metric[i]['metrics'][j]['metric_score'] = macro_specificity(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'micro_MulticlassF1Score':
                micro_MulticlassF1Score = ildv.MulticlassF1Score(average='micro', num_classes=num_classes).to(device)
                micro_MulticlassF1Score.update(output, label)
                metric[i]['metrics'][j]['metric_score'] = micro_MulticlassF1Score.compute().item()
            elif metric[i]['metrics'][j]['metric_name'] == 'macro_MulticlassF1Score':
                macro_MulticlassF1Score = ildv.MulticlassF1Score(average='macro', num_classes=num_classes).to(device)
                macro_MulticlassF1Score.update(output, label)
                metric[i]['metrics'][j]['metric_score'] = macro_MulticlassF1Score.compute().item()
            elif metric[i]['metrics'][j]['metric_name'] == 'multiclassBalancedAccuracy':
                multiclassBalancedAccuracy = ildv.MulticlassBalancedAccuracy(num_classes).to(device)
                multiclassBalancedAccuracy.update(output, label)
                metric[i]['metrics'][j]['metric_score'] = multiclassBalancedAccuracy.compute().item()
            elif metric[i]['metrics'][j]['metric_name'] == 'multiclassOptimizedPrecision':
                multiclassOptimizedPrecision = ildv.MulticlassOptimizedPrecision(num_classes).to(device)
                multiclassOptimizedPrecision.update(output, label)
                metric[i]['metrics'][j]['metric_score'] = multiclassOptimizedPrecision.compute().item()
            elif metric[i]['metrics'][j]['metric_name'] == 'mcc':
                mcc = MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)
                metric[i]['metrics'][j]['metric_score'] = mcc(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'macro_auc':
                macro_auc = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
                metric[i]['metrics'][j]['metric_score'] = macro_auc(output, label).item()
            elif metric[i]['metrics'][j]['metric_name'] == 'gm':
                macro_recall = Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
                x = macro_recall(output, label).item()
                macro_specificity = Specificity(task='multiclass', average='macro', num_classes=num_classes).to(device)
                macro_specificity.update(output, label)
                y = macro_specificity(output, label).item()
                gm = math.sqrt(x * y)
                metric[i]['metrics'][j]['metric_score'] = gm
            # elif metric[i]['metrics'][j]['metric_name'] == 'ERR':
            #     micro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
            #     x = micro_accuracy(output, label).item()
            #     metric[i]['metrics'][j]['metric_score'] = 1-x
#-------------------------------------------------------------------------------------------------------
    # print(metric)
#[{'value similarity':{'acc':100,}}]
    # if metric.metrics.metric_name == 'micro_accuracy':
    #     micro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
    #     metric.metrics.metric_score = micro_accuracy(output, label).item()
    # if metric.metrics.metric_name == 'macro_accuracy':
    #     macro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
    #     metric.metrics.metric_score = macro_accuracy(output, label).item()
    # if metric.metrics.metric_name ==
    # scores = {}
    # micro_precision = Precision(task="multiclass", average='micro', num_classes=num_classes).to(device)
    # macro_precision = Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
    # scores['micro_precision'] = micro_precision(output, label).item()
    # scores['macro_precision'] = macro_precision(output, label).item()
    # micro_recall = Recall(task='multiclass', average='micro', num_classes=num_classes).to(device)
    # macro_recall = Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
    # scores['micro_recall'] = micro_recall(output, label).item()
    # scores['macro_recall'] = macro_recall(output, label).item()
    # micro_specificity = Specificity(task='multiclass', average='micro', num_classes=num_classes).to(device)
    # micro_specificity.update(output, label)
    # macro_specificity = Specificity(task='multiclass', average='macro', num_classes=num_classes).to(device)
    # macro_specificity.update(output, label)
    # scores['micro_specificity'] = micro_specificity(output, label).item()
    # scores['macro_specificity'] = macro_specificity(output, label).item()
    # micro_MulticlassF1Score = ildv.MulticlassF1Score(average='micro', num_classes=num_classes).to(device)
    # micro_MulticlassF1Score.update(output, label)
    # scores['micro_MulticlassF1Score'] = micro_MulticlassF1Score.compute().item()
    # macro_MulticlassF1Score = ildv.MulticlassF1Score(average='macro', num_classes=num_classes).to(device)
    # macro_MulticlassF1Score.update(output, label)
    # scores['macro_MulticlassF1Score'] = macro_MulticlassF1Score.compute().item()
    # multiclassBalancedAccuracy = ildv.MulticlassBalancedAccuracy(num_classes).to(device)
    # multiclassBalancedAccuracy.update(output, label)
    # scores['MulticlassBalancedAccuracy'] = multiclassBalancedAccuracy.compute().item()
    # multiclassOptimizedPrecision = ildv.MulticlassOptimizedPrecision(num_classes).to(device)
    # multiclassOptimizedPrecision.update(output, label)
    # scores['MulticlassOptimizedPrecision'] = multiclassOptimizedPrecision.compute().item()
    # mcc = MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)
    # scores['Mcc'] = mcc(output, label).item()
    # print(scores)

    # micro_auc = MulticlassAUROC(num_classes = num_classes, average='micro')
    # scores['micro_auc'] = micro_auc(output, label).item()
    # macro_auc = MulticlassAUROC(num_classes = num_classes, average='macro')
    # scores['macro_auc'] = macro_auc(output, label).item()
    # gm = math.sqrt(scores['macro_recall'] * scores['macro_specificity'])
    # scores['gm'] = gm
    # ERR = 1 - scores['micro_accuracy']
    # scores['ERR'] = ERR
    # for key in scores:
    #     metric_value.append(share.ans(key, scores[key]))
    return 2, metric, "success"
    # macro_npv, micro_npv = NPV(num, total, num_classes)
    # scores['macro_npv'] = macro_npv
    # scores['micro_npv'] = macro_npv
    # macro_Mcc = Macro_Mcc(num, total, all)
    # micro_Fbeta = metrics.fbeta_score(label, output, average='micro', beta=1)
    # macro_Fbeta = metrics.fbeta_score(label, output, average='macro', beta=1)
    # macro_gm = Macro_Gm(num, total, all)
    # macro_npv, micro_npv = NPV(num, total, all)
    # ERR = 1 - micro_accuracy
    # micro_fpr, macro_fpr = FPR(num, total, all)
    # micro_fnr, macro_fnr = FNR(num, total, all)
    # micro_for, macro_for = FOR(num, total, all)
    # macro_BA = (macro_recall + macro_specificity) / 2
    # macro_op = OP(num, total, all, micro_accuracy)

# if __name__ == '__main__':
    # metric = [
    #     {
    #         "perspective_name": "Value Similarity",
    #         "metrics": [
    #             {
    #                 "metric_name": "micro_accuracy",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "micro_precision",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "micro_recall",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "micro_specificity",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "mcc",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "micro_MulticlassF1Score",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "gm",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "macro_auc",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "macro_accuracy",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "macro_precision",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "macro_recall",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "macro_specificity",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "macro_MulticlassF1Score",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "multiclassBalancedAccuracy",
    #                 "metric_score": 0
    #             },
    #             {
    #                 "metric_name": "multiclassOptimizedPrecision",
    #                 "metric_score": 0
    #             }
    #         ]
    #     },
    #     {
    #         "perspective_name": "value difference",
    #         "metrics": [
    #             {
    #                 "metric_name": "ERR",
    #                 "metric_score": 0
    #             }
    #         ]
    #     }
    # ]

    # print(len(metric))
    # print(len(metric[0]['metrics']))
    # print(len(metric[1]['metrics']))
    # class_rightness_imagenet("/nfs3-p1/hjc/datasets/imagenet1k/val", "ImageNet", "/home/xjb/code/vipa-model/classfication/Modle.py","/nfs3-p1/hjc/pretraind_models/checkpoints/resnet18-f37072fd.pth", 1000, metric)





