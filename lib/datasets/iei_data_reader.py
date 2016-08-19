# --------------------------------------------------------
# Read image extraction data
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
from iei_eval import iei_eval, load_truth_from_xmlnode
from fast_rcnn.config import cfg

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, help="Path to input data")
    args = parser.parse_args()
    return args


class iei_data_reader(imdb):
    def __init__(self, image_set, data_path):
        imdb.__init__(self, str(image_set))
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, data_path)
        self._classes = ('__background__', # always index 0
                         'sign')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._label_ext = '_labeled.xml'
        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self.set_proposal_method('selective_search')
        self.competition_mode(False)

        # Put any IEI-specific config options here
        self.config = {'cleanup' : True}

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # def default_roidb(self):
    #     raise NotImplementedError

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print "cache file is %s" % cache_file

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_iei_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
        self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        roidb = self._load_selective_search_roidb(gt_roidb)

        # write cache file
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)

        return roidb

    def _load_iei_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in IEI's image
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + self._label_ext)

        input_node = ET.parse(filename).getroot()
        objs = load_truth_from_xmlnode(input_node)
        num_objs = len(objs)

        # todo filter here if we want to do something like excluding difficult images
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            geometry = obj[0]
            poly = loads(geometry.ExportToWkt())

            # todo ensure that these are 0-based
            xmin, ymin, xmax, ymax = poly.bounds

            # Make pixel indexes 0-based
            boxes[ix, :] = [xmin-1, ymin-1, xmax-1, ymax-1]
            klass = self._class_to_ind['sign'] # todo use sign class name in obj[1]

            gt_classes[ix] = klass
            overlaps[ix, klass] = 1.0
            seg_areas[ix] = (xmax - xmin + 1) * (ymax - ymin + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _load_selective_search_roidb(self, gt_roidb):
        """
        Load ROI proposals
        These can be downloaded using data/scripts/fetch_selective_search_data.sh
        """
        ##################
        # todo if we want to use proposals, read in MSER results here
        raise NotImplementedError
        ##################





    def _get_iei_results_file_template(self):
        # <data path>/Main/test_sign.txt
        filename = self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._data_path,
            'results',
            'IEI',
            'Main',
            filename)
        return path

    def _write_iei_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} IEI results file'.format(cls)
            filename = self._get_iei_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # revert back to 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            '{:s}' + self._label_ext)
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_iei_results_file_template().format(cls)
            rec, prec, ap = iei_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_iei_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_iei_results_file_template().format(cls)
                os.remove(filename)



if __name__ == '__main__':
    args = parse_args()
    reader = iei_data_reader('trainval', args.input_dir)
    result = reader.roidb
