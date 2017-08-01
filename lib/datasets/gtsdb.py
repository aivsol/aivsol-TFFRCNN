# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
# TODO
#import utils.cython_bbox
import cPickle
import subprocess
import uuid

from ..utils import cython_bbox
from ..fast_rcnn.config import cfg
from .imdb import imdb
from .imdb import ROOT_DIR
import ds_utils

class gtsdb(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'gtsdb' + '_' + image_set)
        self._year = '2009'
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'speedlimit-20', 'speedlimit-30', 'speedlimit-50', 'speedlimit-60',
                         'speedlimit-70', 'speedlimit-80', 'restrict-end-80', 'speedlimit-100', 'speedlimit-120',
                         'no-overtake', 'no-overtake-truck', 'priority-next-intersect', 'priority-road',
                         'giveaway', 'stop', 'no-traffic-bothways', 'no-truck', 'no-entry', 'danger',
                         'bend-left', 'bend-right','bend', 'uneven-road', 'slippery-road', 'road-narrow',
                         'construction', 'traffic-signal', 'pedestrian-crossing', 'school-crossing', 'cycle-crossing',
                         'snow', 'animals', 'restriction-ends', 'go-right', 'go-left', 'go-straight',
                         'go-right-straight', 'go-left-straight', 'keep-right', 'keep-left', 'roundabout',
                         'restrict-ends-overtaking', 'restrict-ends-overtaking-truck')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.ppm'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # GTSDB specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            if self._image_set != 'test':
                image_index_collection = [x.split(';')[0].split('.')[0] for x in f.readlines()]
                image_index = list(set(image_index_collection))
            else:
                image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'GTSDB')

    def gt_roidb(self):

        gt_roidb = [self._load_gtsdb_annotation(index)
                    for index in self.image_index]
        return gt_roidb

    def _load_gtsdb_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the INRIA VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', 'gt.txt')
        bb = []
        with open(filename) as f:
            for x in f.readlines():
                gt_ind = x.split(';')[0].split('.')[0]
                if index == gt_ind:
                    bb.append(x)
        num_objs = len(bb)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(bb):
            bbox = obj.split(';')
            # Make pixel indexes 0-based
            x1 = float(bbox[1]) - 1
            y1 = float(bbox[2]) - 1
            x2 = float(bbox[3]) - 1
            y2 = float(bbox[4]) - 1
            cls = int(bbox[5]) + 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path


    def evaluate_detections(self, all_boxes, output_dir):
        # Get GT
        gt = self.gt_roidb()
        aps = []
        arcs = []
        atps = []
        afps = []
        afns = []
        # Looping over classes
        for cls_index in xrange(1, len(self._classes)):
            cls_detections = all_boxes[cls_index]
            tp_total = 0; fp_total = 0; fn_total = 0
            # Looping over images within the context of a particular class
            for image_index in range(len(cls_detections)):
                # Retrieve detections for this particular image and class
                img_cls_detections = cls_detections[image_index]
                # True Positive, False Positive and False Negative
                tp = 0; fp = 0; fn = 0
                # Retrieve GT for this particular image and class
                gt_boxes = gt[image_index]['boxes']
                gt_cls = gt[image_index]['gt_classes']
                gt_boxes_cls = gt_boxes[np.where(gt_cls==cls_index)[0]]

                # Keep track of GT which are matched by a detection
                markings = [False] * len(gt_boxes_cls)
                # Let's go over each detection
                for det in img_cls_detections:
                    bbox = det[:4]

                    tp_found = False
                    for idx, gt_box in enumerate(gt_boxes_cls):
                        # Continue if the GT is already matched
                        if markings[idx] is True:
                            continue
                        iou = self.iou(bbox, gt_box)
                        if iou  > 0.5:
                            tp+=1 # If the detection matches any gt, tp++
                            tp_found = True
                            markings[idx] = True
                            break # Not sure if we should break
                    if not tp_found:
                        fp+=1 # If the detection doesn't match any gt, fp++
                # GT not matched by any detection are added to fn
                fn = markings.count(False)
                tp_total += tp
                fp_total += fp
                fn_total +=fn

            if tp_total > 0:
                prec = tp_total/float((tp_total + fp_total))
                recall = tp_total/float((tp_total + fn_total))
                print ('-'*20)
                print('Precision for {} = {:.4f}'.format(
                                                self._classes[cls_index],
                                                prec))
                print ('Recall for {} = {:.4f}'.format(
                                                self._classes[cls_index],
                                                recall))
                print 'TP: ', tp_total, ' FP: ', fp_total, ' FN: ', fn_total
                aps += [prec]
                arcs += [recall]
                atps += [tp_total]
                afps += [fp_total]
                afns += [fn_total]

        print ('-'*20)
        print ('Average Precision: {:.4f}'.format(np.mean(aps)))
        print ('Average Recall: {:.4f}'.format(np.mean(arcs)))
        print ('True Positives: {}'.format(np.sum(atps)))
        print ('False Positives: {}'.format(np.sum(afps)))
        print ('False Negatives: {}'.format(np.sum(afns)))

    def iou(self, boxA, boxB):
	ixmin = np.maximum(boxA[0], boxB[0])
	iymin = np.maximum(boxA[1], boxB[1])
	ixmax = np.minimum(boxA[2], boxB[2])
	iymax = np.minimum(boxA[3], boxB[3])
	iw = np.maximum(ixmax - ixmin + 1., 0.)
	ih = np.maximum(iymax - iymin + 1., 0.)
	inters = iw * ih

	# union
	uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
	       (boxA[2] - boxA[0] + 1.) * (boxA[3] - boxA[1] + 1.) - inters)

	overlaps = inters / uni

        return overlaps

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.gtsdb import gtsdb
    d = gtsdb('train', '')
    res = d.roidb
    from IPython import embed; embed()
