import numpy as np
from scipy.spatial.distance import cdist
import cv2
from ultralytics.yolo.utils.plotting import Annotator
from random import choice

# List of colors for annotation boxes
colors = [(97, 52, 235), (52, 235, 110), (235, 168, 52), (216, 52, 235), (168, 32, 62), (8, 158, 75),
          (49, 88, 117), (143, 74, 199), (224, 121, 173), (0, 0, 0), (121, 121, 121), (71, 143, 81), (111, 252, 3),
          (14, 143, 117), (106, 132, 217), (94, 52, 133), (143, 44, 150), (184, 46, 119), (207, 192, 81), (111, 122, 171)]


def pred_model(results):
    # Extract bounding boxes and class labels from model results
    box_list = []
    cls_list = []
    for res in results:
        boxes = res.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])
            cls_list.append(cls)
            box_list.append(box.xyxy[0].astype(int))
    return box_list, cls_list

def calculate_iou(box1, box2):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    intersection_area = (
        max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) *
        max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    )
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def predict_loop(path_img, mother_cls_list,model_cul, model_leg, model_NO_leg, model_plate):
    box_cul = []
    cls_cul = []
    box_leg = []
    cls_leg = []
    box_no_leg = []
    cls_no_leg = []
    box_plate = []
    cls_plate = []
    for class_id in set(mother_cls_list):
        if class_id == 0:
            results_cul = model_cul.predict(path_img, agnostic_nms=True)
            box_cul, cls_cul = pred_model(results_cul)
        if class_id == 1:
            results_leg = model_leg.predict(path_img)
            box_leg, cls_leg = pred_model(results_leg)
        if class_id == 2:
            results_no_leg = model_NO_leg.predict(path_img)
            box_no_leg, cls_no_leg = pred_model(results_no_leg)
        if class_id == 3:
            results_plate = model_plate.predict(path_img)
            box_plate, cls_plate = pred_model(results_plate)
    return box_cul, cls_cul, box_leg, cls_leg, box_no_leg, cls_no_leg, box_plate, cls_plate

# Порог для значения IoU
iou_threshold = 0.5
# Список для хранения боксов с высоким значением IoU и информации о моделях
high_iou_boxes = []

# Функция для проверки перекрывающихся боксов между уточняющими моделями
def check_high_iou_boxes(box_list1, cls_list1, model1, box_list2, cls_list2, model2, high_iou_boxes, iou_threshold=0.5):
    # Check for high IoU boxes between two sets of bounding boxes from different models
    for box1, cls1 in zip(box_list1, cls_list1):
        for box2, cls2 in zip(box_list2, cls_list2):
            iou = calculate_iou(box1, box2)
            if iou >= iou_threshold:
                high_iou_boxes.append((box1, cls1, model1, box2, cls2, model2))
    return high_iou_boxes






# Функция для нахождения бокса, ближайшего к обоим боксам
def find_nearest_box_to_pair(pair_box, box_list):
    # Find the nearest box in `box_list` to the pair of boxes in `pair_box`
    distances = cdist(pair_box, box_list, 'euclidean')
    sum_distances = np.sum(distances, axis=0)
    nearest_index = np.argmin(sum_distances)
    nearest_box = box_list[nearest_index]
    return nearest_box, nearest_index


def predict_img(img_path, model_mother, model_cul, model_leg, model_NO_leg, model_plate):
    results_mother = model_mother(img_path, conf=0.4)
    mother_bboxes_list, mother_cls_list = pred_model(results_mother)
    box_cul, cls_cul, box_leg, cls_leg,\
        box_no_leg, cls_no_leg, box_plate, cls_plate = predict_loop(img_path, mother_cls_list,
                                                                    model_cul, model_leg, model_NO_leg, model_plate)
    # Порог для значения IoU
    iou_threshold = 0.5
    # Список для хранения боксов с высоким значением IoU и информации о моделях
    high_iou_boxes = []
    # Проверка перекрывающихся боксов между уточняющими моделями
    high_iou_boxes = check_high_iou_boxes(box_cul, cls_cul, 'model_cul', box_leg, cls_leg, 'model_leg', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_cul, cls_cul, 'model_cul', box_no_leg, cls_no_leg, 'model_no_leg', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_cul, cls_cul, 'model_cul', box_plate, cls_plate, 'model_plate', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_leg, cls_leg, 'model_leg', box_cul, cls_cul, 'model_cul', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_leg, cls_leg, 'model_leg', box_no_leg, cls_no_leg, 'model_no_leg', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_leg, cls_leg, 'model_leg', box_plate, cls_plate, 'model_plate', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_no_leg, cls_no_leg, 'model_no_leg', box_cul, cls_cul, 'model_cul', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_no_leg, cls_no_leg, 'model_no_leg', box_leg, cls_leg, 'model_leg', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_no_leg, cls_no_leg, 'model_no_leg', box_plate, cls_plate, 'model_plate', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_plate, cls_plate, 'model_plate', box_cul, cls_cul, 'model_cul', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_plate, cls_plate, 'model_plate', box_leg, cls_leg, 'model_leg', high_iou_boxes)
    high_iou_boxes = check_high_iou_boxes(box_plate, cls_plate, 'model_plate', box_no_leg, cls_no_leg, 'model_no_leg', high_iou_boxes)

    # Словарь соответствия моделей и индексов
    model_index_mapping = {0: 'model_cul', 1: 'model_leg', 2: 'model_no_leg', 3: 'model_plate'}

    # Список для хранения пар боксов с высоким IoU, их наиболее близких боксов из mother_bboxes_list,
    # и моделей, которые не соответствуют наиближайшему боксу
    matched_boxes = []
    print(high_iou_boxes)
    # Нахождение наиболее близкого бокса для каждой пары боксов с высоким IoU
    if high_iou_boxes:
        for box1, cls1, _, box2, cls2, _ in high_iou_boxes:

            pair_box = np.vstack((box1, box2))
            nearest_box, nearest_index = find_nearest_box_to_pair(pair_box, mother_bboxes_list)
            match_cls = model_index_mapping.get(mother_cls_list[nearest_index])
            remove_list = []
            for elem in high_iou_boxes:
                elem = list(elem)
                for ind, el in enumerate(elem.copy()):
                    if el == match_cls:
                        del (elem[ind - 2:ind + 1])
                remove_list.append(elem)
        remove_box_dict = {'model_cul': [box_cul, cls_cul], 'model_leg': [box_leg, cls_leg],
                           'model_no_leg': [box_no_leg, cls_no_leg], 'model_plate': [box_plate, cls_plate]}

        # Создаем новый список для хранения уникальных элементов
        unique_remove_list = []

        # Перебираем элементы исходного списка
        for item in remove_list:
            is_duplicate = False
            for unique_item in unique_remove_list:
                if len(item) != len(unique_item):
                    continue
                is_equal = True
                for a, b in zip(item, unique_item):
                    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                        if not np.array_equal(a, b):
                            is_equal = False
                            break
                    elif a != b:
                        is_equal = False
                        break
                if is_equal:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_remove_list.append(item)


        for elem in unique_remove_list:
            try:
                ind_cls = 0
                for el in remove_box_dict.get(elem[-1])[0]:
                    print(el)
                    if len(el) != len(elem[0]):
                        continue
                    else:
                        for i in range(len(el)):
                            if el[i] != elem[0][i]:
                                continue
                        else:
                            ind_cls +=1
                            break
                print(remove_box_dict.get(elem[-1])[1].pop(ind_cls))
                remove_box_dict.get(elem[-1])[0].pop(ind_cls)
            except:
                print('er')

    img = cv2.imread(img_path)
    annotator = Annotator(img)

    for i, box in enumerate(box_cul):
        r = box
        cls = cls_cul[i]
        annotator.box_label(r, model_cul.names[cls], color=choice(colors))
    for i, box in enumerate(box_leg):
        r = box
        cls = cls_leg[i]
        annotator.box_label(r, model_leg.names[cls], color=choice(colors))
    for i, box in enumerate(box_no_leg):
        r = box
        cls = cls_no_leg[i]
        annotator.box_label(r, model_NO_leg.names[cls], color=choice(colors))
    for i, box in enumerate(box_plate):
        r = box
        cls = cls_plate[i]
        annotator.box_label(r, model_plate.names[cls], color=choice(colors))

    result_image = annotator.result()

    cv2.imwrite(img_path, result_image)
    return img_path