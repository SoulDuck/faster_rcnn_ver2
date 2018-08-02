import os
from scipy.misc import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def next_img_gtboxes(image_idx):
    IMAGE_FORMAT= '.png'
    data_dir='./clutteredMNIST'
    train_name_path = os.path.join(data_dir, 'Names', 'train.txt')
    train_names = [line.rstrip() for line in open(train_name_path, 'r')]
    if image_idx > (len(train_names)-1) :
        image_idx= image_idx % (len(train_names)-1)


    img_path = os.path.join(data_dir, 'Images', train_names[image_idx] + IMAGE_FORMAT)
    annotation_path = os.path.join(data_dir, 'Annotations', train_names[image_idx] + '.txt')
    img = imread(img_path)

    gt_bbox = np.loadtxt(annotation_path, ndmin=2)
    #im_dims = np.array(img.shape[:2]).reshape([1, 2])

    flips = [0, 0]
    flips[0] = np.random.binomial(1, 0.5)
    #img = image_preprocessing.image_preprocessing(img)
    if np.max(img) > 1:
        img = img / 255.
    return img , gt_bbox
def non_maximum_supression( dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]]) # xx1 shape : [19,]
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h #inter shape : [ 19,]
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def draw_rectangles_fastrcnn(img , bboxes , true_classes  , savepath):
    ax = plt.axes()

    bboxes = np.asarray(bboxes)
    bboxes = np.squeeze(bboxes)
    print np.shape(bboxes)
    plt.imshow(img)
    h,w = np.shape(img)
    pos_indices=np.where([true_classes > 0])[1]
    neg_indices = np.where([true_classes == 0])[1]
    print pos_indices
    pos_bboxes = bboxes[pos_indices]
    neg_bboxes = bboxes[neg_indices]


    for box in pos_bboxes :

        x1, y1, x2, y2= box  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        else:
            continue

    for box in neg_bboxes :
        x1, y1, x2, y2= box  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            continue

    plt.savefig(savepath)
    plt.close()






def draw_rectangles(img ,bboxes ,scores , anchors, roi_nms_bbox , savepath , color):
    ax = plt.axes()
    plt.imshow(img)
    h,w=np.shape(img)
    pos_bboxes_indices = np.where([scores >= 0.5])[1]
    neg_bboxes_indices = np.where([scores < 0.5])[1]
    pos_bboxes=bboxes[pos_bboxes_indices]
    pos_bboxes = pos_bboxes[:,1:]
    neg_bboxes = bboxes[neg_bboxes_indices]
    neg_bboxes = neg_bboxes[:,1:]

    # DRAW POS BBOX
    for box in pos_bboxes:

        x1, y1, x2, y2= box  # x1 ,y1 ,x2 ,y2
        if x1 >0 and y1 >0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h :
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        else:
            continue
    plt.savefig(savepath.replace('roi' , 'pos_roi'))
    plt.close()

    # DRAW NEG BBOX
    ax = plt.axes()
    plt.imshow(img)
    h,w=np.shape(img)
    for box in neg_bboxes:
        x1, y1, x2, y2 = box  # x1 ,y1 ,x2 ,y2
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            continue

    plt.savefig(savepath.replace('roi', 'neg_roi'))
    plt.close()
    # DRAW ANCHOR BBOX
    ax = plt.axes()
    plt.imshow(img)
    h,w=np.shape(img)
    for box in anchors:

        x1, y1, x2, y2 = box  # x1 ,y1 ,x2 ,y2
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        else:
            continue
    plt.savefig(savepath.replace('roi', 'anchor'))
    plt.close()

    # DRAW Non Maximun Surpress
    if not roi_nms_bbox is None :
        ax = plt.axes()
        plt.imshow(img)
        h,w=np.shape(img)
        for box in roi_nms_bbox :

            x1, y1, x2, y2 = box  # x1 ,y1 ,x2 ,y2
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and x2 > x1 and y2 > y1 and w > x2 and y2 < h:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='y', facecolor='none')
                ax.add_patch(rect)
            else:
                continue
        plt.savefig(savepath.replace('roi', 'nms_roi'))
        plt.close()



if '__main__' == __name__:
    img , gt_boxes =next_img_gtboxes(image_idx=1)
    ax=plt.axes()
    for box in gt_boxes:
        x1, y1, x2, y2, label = box  # x1 ,y1 ,x2 ,y2
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.imshow(img)
    plt.show()


