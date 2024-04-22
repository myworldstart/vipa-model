import importlib
import sys
sys.path.append("/home/xjb/code/vipa-model/")
sys.path.append("/home/xjb/code/vipa-model/cocoLRPapi-master/PythonAPI/pycocotools")
sys.path.append("/home/xjb/code/vipa-model/cocoLRPapi-master/PythonAPI")
print(sys.path)
import torch
import loaders
from metrics import ilpl
from tqdm import tqdm
import time

# sys.path.append("/home/xjb/code/project1/Obj_detection/")
# sys.path.append("/home/xjb/code/project1/loaders/")
torch.multiprocessing.set_sharing_strategy('file_system')
def test(data, data_path, data_ann_path, model_path, parameter_path):
    if data == "COCO":
        test_coco(data_path, data_ann_path, model_path, parameter_path)
    elif data == "imagenet":
        pass



def test_coco(data_path, data_ann_path, model_path, parameter_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Moduel = importlib.import_module("Modle")
    model = Moduel.get_model()
    #model.load_state_dict(torch.load(parameter_path))
    test_loader = loaders.load_detection_data(data_name="COCO",
                                              data_dir=data_path,
                                              data_ann_path=data_ann_path,
                                              data_type='val')
    evaluates = [
        ilpl.MeanAveragePrecision()
    ]
    cocoeval = ilpl.COCOMetrics(data_ann_path)
    model.eval()
    model.to(device)
    res_list = []
    for i, samples in enumerate(tqdm(test_loader)):
        inputs, labels = samples

        inputs = list(img.to(device) for img in inputs)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
        # print(torch.tensor(inputs).shape)
        # print(labels[0])

        with torch.no_grad():
            outputs = model(inputs)

        outputs = [{k: v for k, v in t.items()} for t in outputs]

        # ================== prepare for LRP Error in cocoeval.py ==================

        def prepare_for_coco_detection(predictions):
            coco_results = []
            for original_id, prediction in predictions.items():
                if len(prediction) == 0:
                    continue

                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()
                scores = prediction["scores"].tolist()
                labels = prediction["labels"].tolist()

                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            return coco_results

        def convert_to_xywh(boxes):
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

        res = {target["image_id"].item(): output for target, output in zip(labels, outputs)}
        results = prepare_for_coco_detection(res)
        res_list.extend(results)

        # ================== prepare for LRP Error in cocoeval.py ==================

        # print(results)
        # print('-' * 20)

        for evaluate in evaluates:
            evaluate.update(outputs, labels)

    state, stateeval= cocoeval.compute(res_list)
    for i in range(len(state)):
        print(state[i])
    print('moLRP' + stateeval['moLRP'])
    # #-----------------------------LRP--------------------------------
    # output_json_file = 'predictions.json'
    # with open(output_json_file, 'w') as json_file:
    #     json.dump(res_list, json_file)
    # cocogt = COCO(data_ann_path)

    scores = []
    for evaluate in evaluates:
        score = evaluate.compute()
        scores.append(score)

    print(scores)
    return scores

if __name__ == '__main__':
    test("COCO",'/datasets/COCO2017/images/val','/datasets/COCO2017/annotations/instances_val2017.json', "ssd","'/nfs3-p1/hjc/pretraind_models/checkpoints/ssd300_vgg16_coco-b556d3b4.pth'")





