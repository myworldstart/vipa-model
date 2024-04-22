import importlib
import sys

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
from classfication.Modle import Net
from sklearn import metrics
from torchmetrics.classification import MulticlassSpecificity, MulticlassAccuracy, MulticlassMatthewsCorrCoef, \
    MulticlassAUROC
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
def class_rightness_imagenet(data_path, data, model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Moduel = importlib.import_module("Modle")
    # model = Moduel.get_model()
    model = models.load_model(model_name="ResNet18", num_classes=num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # model.load_state_dict(torch.load(parameter_path))
    total = 0
    outputss = []
    labelss = []
    test_loader = loaders.load_data(data_name=data, data_dir=data_path, data_type='test')
    for i, samples in enumerate(tqdm(test_loader)):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        outputss.append(outputs)
        labelss.append(labels)

    output = torch.cat(outputss,dim=0)
    label = torch.cat(labelss, dim=0)
    # total = len(label)
    # for i in range(len(label)):
    #     y_true = label[i]
    #     if(output[i] == y_true):
    #         num[label][0] = num[label][0] + 1
    #     else:
    #         num[label][1] += 1
    #         num[output[i]][2] += 1
    scores = {}
    micro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
    macro_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
    scores['micro_accuracy'] = micro_accuracy(output, label).item()
    scores['macro_accuracy'] = macro_accuracy(output, label).item()
    micro_precision = Precision(task="multiclass", average='micro', num_classes=num_classes)
    macro_precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
    scores['micro_precision'] = micro_precision(output, label).item()
    scores['macro_precision'] = macro_precision(output, label).item()
    micro_recall = Recall(task='multiclass', average='micro', num_classes=num_classes)
    macro_recall = Recall(task='multiclass', average='macro', num_classes=num_classes)
    scores['micro_recall'] = micro_recall(output, label).item()
    scores['macro_recall'] = macro_recall(output, label).item()
    micro_specificity = Specificity(task='multiclass', average='micro', num_classes=num_classes)
    micro_specificity.update(output, label)
    macro_specificity = Specificity(task='multiclass', average='macro', num_classes=num_classes)
    macro_specificity.update(output, label)
    scores['micro_specificity'] = micro_specificity(output, label).item()
    scores['macro_specificity'] = macro_specificity(output, label).item()
    micro_MulticlassF1Score = ildv.MulticlassF1Score(average='micro', num_classes=num_classes)
    micro_MulticlassF1Score.update(output, label)
    scores['micro_MulticlassF1Score'] = micro_MulticlassF1Score.compute().item()
    macro_MulticlassF1Score = ildv.MulticlassF1Score(average='macro', num_classes=num_classes)
    macro_MulticlassF1Score.update(output, label)
    scores['macro_MulticlassF1Score'] = macro_MulticlassF1Score.compute().item()
    multiclassBalancedAccuracy = ildv.MulticlassBalancedAccuracy(num_classes)
    multiclassBalancedAccuracy.update(output, label)
    scores['MulticlassBalancedAccuracy'] = multiclassBalancedAccuracy.compute().item()
    multiclassOptimizedPrecision = ildv.MulticlassOptimizedPrecision(num_classes)
    multiclassOptimizedPrecision.update(output, label)
    scores['MulticlassOptimizedPrecision'] = multiclassOptimizedPrecision.compute().item()
    mcc = MulticlassMatthewsCorrCoef(num_classes=num_classes)
    scores['Mcc'] = mcc(output, label).item()


    # micro_auc = MulticlassAUROC(num_classes = num_classes, average='micro')
    # scores['micro_auc'] = micro_auc(output, label).item()
    macro_auc = MulticlassAUROC(num_classes = num_classes, average='macro')
    scores['macro_auc'] = macro_auc(output, label).item()
    gm = math.sqrt(scores['macro_recall'] * scores['macro_specificity'])
    scores['gm'] = gm
    ERR = 1 - scores['micro_accuracy']
    scores['ERR'] = ERR
    print(scores)
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

if __name__ == '__main__':
    class_rightness_imagenet("/nfs3-p1/hjc/datasets/imagenet1k/val", "ImageNet", "/nfs3-p1/hjc/pretraind_models/checkpoints/resnet18-f37072fd.pth", 1000)





