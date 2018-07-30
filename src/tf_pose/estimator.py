import logging

import src.tf_pose.slidingwindow as sw

from src.tf_pose.pafprocess import pafprocess
import cv2
import numpy as np
import tensorflow as tf
import time

import src.tf_pose.common as common
from src.tf_pose.common import CocoPart
from src.tf_pose.tensblur.smoother import Smoother

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))
                #printk_queue[2],peak_queue[0]('====>flag:'+str(flag))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:
    # TODO : multi-scale

    def __init__(self, graph_path, target_size=(320, 240)):
        self.target_size = target_size

        # load graph
        logger.info('loading graph from %s(default size=%dx%d)' % (graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph)

        # for op in self.graph.get_operations():
        #     print(op.name)
        # for ts in [n.name for n in tf.get_default_graph().as_graph_def().node]:
        #     print(ts)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.tensor_heatMat = self.tensor_output[:, :, :, :19]
        self.tensor_pafMat = self.tensor_output[:, :, :, 19:]
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2, ), name='upsample_size')
        self.tensor_heatMat_up = tf.image.resize_area(self.tensor_output[:, :, :, :19], self.upsample_size, align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.tensor_output[:, :, :, 19:], self.upsample_size, align_corners=False, name='upsample_pafmat')
        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat, tf.zeros_like(gaussian_heatMat))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer(
            [v for v in tf.global_variables() if
             v.name.split(':')[0] in [x.decode('utf-8') for x in self.persistent_sess.run(tf.report_uninitialized_variables())]
             ])
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1], target_size[0]]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 2, target_size[0] // 2]
            }
        )
        self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)],
                self.upsample_size: [target_size[1] // 4, target_size[0] // 4]
            }
        )
        #self.peak_queue = []
        #self.pose_flag = 0

    def __del__(self):
        # self.persistent_sess.close()
        pass

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2**8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    @staticmethod
    def draw_humans(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        peaks_flag = [3,4,6,7]
        human_flag = 0
        result_txt = ''
        distance = ''
        spead = ''
        result = 0
        for human in humans:
            peaks = []
            flag = 0
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    if i<8:
                        if i == peaks_flag[flag]:
                            peaks.append([-1])
                            flag +=1
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                #cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)
                cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=-1)
                #cv2.putText(npimg,str(i),center,0, 5e-3 * 100, (0,0,0),1)
                if i<8:
                    if i == peaks_flag[flag]:
                        peaks.append(tuple(center))
                        #print('peaks:'+str(peaks[-1]))
                        flag +=1
            #if 0 in human.body_parts.keys():
            #    cv2.putText(npimg,str(human_flag),(centers[0][0]-5,centers[0][1]-15),0, 0.7, (0,255,0),2)
            if 1 in human.body_parts.keys(): 
                result,distance,spead = getJudge(peaks,tuple(centers[1]))
            
            if result == 1:
                #print('left')
                result_txt = 'Swing_right'
                #cv2.putText(npimg,'Swing_left', (5,5), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,255,0], 1)
            elif result == 2:
                #print('right')
                result_txt = 'Swing_left'
                #cv2.putText(npimg, 'Swing_right', (20,20), cv2.FONT_HERSHEY_PLAIN, 1.2, [0,255,0], 1)
            human_flag +=1
            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                #npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

        return npimg,result_txt,str(distance),spead
    """
    def _getJudge(self,all_peak):
        self.peak_queue.append(all_peak)
        self.pose_flag
        peaks = []
        result = 0
        if len(self.peak_queue) >=3:
            result = self._Judgment(self.peak_queue[2],self.peak_queue[0])
            print('result:'+str(result))
            self.peak_queue.pop(0)
            self.peak_queue.pop(0)
            if result == self.pose_flag:
                self.pose_flag = 0
                return result
            else:
                pose_flag = result
                return 0
        
        

    def _Judgment(new_peaks,old_peaks):
        new_peaks = np.array(new_peaks)
        old_peaks = np.array(old_peaks)
        right_arm1 = new_peaks[0]
        right_hand1 = new_peaks[1]
        left_arm1 = new_peaks[2]
        left_hand1 = new_peaks[3]

        right_arm2 = old_peaks[0]
        right_hand2 = old_peaks[1]
        left_arm2 = old_peaks[2]
        left_hand2 = old_peaks[3]
        #right
        if not right_hand1 == -1 and not right_hand2 == -1:
            k = (right_hand1[1] - right_hand2[1]) / (right_hand1[0] - right_hand2[0])
            distance = int(((right_hand1[1] - right_hand2[1]) ** 2 + (right_hand1[0] - right_hand2[0]) ** 2) ** 0.5)
        elif not right_arm1 == -1 and not right_arm2 == -1:
            k = (right_arm1[1] - right_arm2[1]) / (right_arm1[0] - right_arm2[0])
            distance = int(((right_arm1[1] - right_arm2[1]) ** 2 + (right_arm1[0] - right_arm2[0]) ** 2) ** 0.5)
        else:
            return 0

        if k > 0 and distance > 1:  # left
            return 1
        if k < 0 and distance > 1:  # right
            return 2
    """       




    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3),
                    dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros((max(self.target_size[1], npimg.shape[0]), max(self.target_size[0], npimg.shape[1]), 3), dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w*ratio_x-.5), 0)
        y = max(int(h*ratio_y-.5), 0)
        cropped = npimg[y:y+target_h, x:x+target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y+cropped_h, copy_x:copy_x+cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, resize_to_default=True, upsample_size=1.0):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]
        peaks, heatMat_up, pafMat_up = self.persistent_sess.run([self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
            self.tensor_image: [img], self.upsample_size: upsample_size
        })
        peaks = peaks[0]
        self.heatMat = heatMat_up[0]
        self.pafMat = pafMat_up[0]
        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        logger.debug('estimate time=%.5f' % (time.time() - t))
        return humans


peak_queue = []
center_queue = []
pose_flag = 0
Distance = 0.0
def getJudge(all_peak,center):
    global peak_queue
    global pose_flag
    global Distance
    global center_queue
    peak_queue.append(all_peak)
    center_queue.append(center)
    #print('===>peak_queue:'+str(peak_queue))
    peaks = []
    result = 0
    if len(peak_queue) >=3:
        result = Judgment(peak_queue[2],peak_queue[0])
        print('result:'+str(result))
        peak_queue.pop(0)
        peak_queue.pop(0)
    spead = ''
    if len(center_queue) >=3:
        #distance
        distance,spead = get_distance(center_queue[2],center_queue[1])
        Distance += float(distance)

    if result == pose_flag:
        pose_flag = 0
        return result,Distance,spead
    else:
        pose_flag = result
        return 0,Distance,spead


def Judgment(new_peaks,old_peaks):
    new_peaks = np.array(new_peaks)
    old_peaks = np.array(old_peaks)
    
    right_arm1 = new_peaks[0]
    right_hand1 = new_peaks[1]
    left_arm1 = new_peaks[2]
    left_hand1 = new_peaks[3]

    right_arm2 = old_peaks[0]
    right_hand2 = old_peaks[1]
    left_arm2 = old_peaks[2]
    left_hand2 = old_peaks[3]
    #right
    #if type(right_hand2)==np.ndarray:
    #    print(len(right_hand2))
    #    right_hand2 = tuple(right_hand2)
    #print(right_hand2)
    #print(isinstance(right_hand1,int))
    #print(isinstance(right_hand2,int))
    #rh1 = not (isinstance((not right_hand1 == -1),bool) and right_hand1 == -1)
    #rh2 = not (isinstance((not right_hand2 == -1),bool) and right_hand2 == -1)
    #ra1 = not (isinstance((not right_arm1 == -1),bool) and right_arm1 == -1)
    #ra2 = not (isinstance((not right_arm2 == -1),bool) and right_arm2 == -1)
    #print(list([rh1,rh2,ra1,ra3]))
    #if not isinstance(right_hand1,int) and not isinstance(right_hand2,int):
    if len(right_hand1) > 1 and len(right_hand2) > 1:
        k = (right_hand1[1] - right_hand2[1]) / (right_hand1[0] - right_hand2[0]) if not right_hand1[0] - right_hand2[0] == 0 else 0
        distance = int(((right_hand1[1] - right_hand2[1]) ** 2 + (right_hand1[0] - right_hand2[0]) ** 2) ** 0.5)
    #elif not isinstance(right_arm1,int) and not isinstance(right_arm2,int):
    elif len(right_arm1) >1 and len(right_arm2) >1:
        k = (right_arm1[1] - right_arm2[1]) / (right_arm1[0] - right_arm2[0]) if not right_arm1[0] - right_arm2[0] == 0 else 0
        distance = int(((right_arm1[1] - right_arm2[1]) ** 2 + (right_arm1[0] - right_arm2[0]) ** 2) ** 0.5)
    elif len(left_hand1) >1 and len(left_hand2) >1:
        k = (left_hand1[1] - left_hand2[1]) / (left_hand1[0] - left_hand2[0]) if not left_hand1[0] - left_hand2[0] == 0 else 0
        distance = int(((left_hand1[1] - left_hand2[1]) ** 2 + (left_hand1[0] - left_hand2[0]) ** 2) ** 0.5)
    elif len(left_arm1) >1 and len(left_arm2) >1:
        k = (left_arm1[1] - left_arm2[1]) / (left_arm1[0] - left_arm2[0]) if not left_arm1[0] - left_arm2[0] == 0 else 0
        distance = int(((left_arm1[1] - left_arm2[1]) ** 2 + (left_arm1[0] - left_arm2[0]) ** 2) ** 0.5)
    else:
        return 0
    print('>>> k='+str(k)+' >>>distance='+str(distance)) 
    if k > 0 and distance > 5:  # left
        return 1
    if k < 0 and distance > 5:  # right
        return 2
    else:
        return 0


def get_distance(center2,center0):
    y1 = center2[1] 
    s = center2[0] - center0[0]
    k = 1.17
    w1 = 1828 / 663 #2.76
    w2 = 1828 / 238 
    w_distance = s - k * (y1 - 336) * s / 663 * w1
    h_distance = (center2[1] - center0[1]) * w2
    all_distance = int(w_distance ** 2 + h_distance ** 2) ** 0.5
    spead = all_distance / (2/25)
    print('==================distance:'+str(all_distance))
    print('==================spead:'+str(spead))
    all_distance = '%.2f' %(all_distance/100)
    spead = '%.2f' %(spead/100)
    return all_distance,str(spead)



if __name__ == '__main__':
    import pickle
    f = open('./etcs/heatpaf1.pkl', 'rb')
    data = pickle.load(f)
    logger.info('size={}'.format(data['heatMat'].shape))
    f.close()

    t = time.time()
    humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
    dt = time.time() - t; t = time.time()
    logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
