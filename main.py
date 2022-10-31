# =======================================================
# Project Name    : Multi People Tracking on Edge
# File Name       : main.py
# Auther          : Taro Ishihara
# =======================================================

import degirum as dg
import numpy as np
import mytools, cv2
from pathlib import Path
import IPython.display

from byte_tracker import BYTETracker

np.set_printoptions(suppress=True)

line_start = (250, 0)
line_end = (250, 500)

left = 0
right = 0
top = 0
bottom = 0

class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def intersect(a, b, c, d):
    s = (a[0] - b[0]) * (c[1] - a[1]) - (a[1] - b[1]) * (c[0] - a[0])
    t = (a[0] - b[0]) * (d[1] - a[1]) - (a[1] - b[1]) * (d[0] - a[0])
    if s * t > 0:
        return False
    s = (c[0] - d[0]) * (a[1] - c[1]) - (c[1] - d[1]) * (a[0] - c[0])
    t = (c[0] - d[0]) * (b[1] - c[1]) - (c[1] - d[1]) * (b[0] - c[0])
    if s * t > 0:
        return False
    return True

inference_option = 1

zoo = mytools.connect_model_zoo(inference_option)

model_list = zoo.list_models()
print(model_list)

# load object detection model
model = zoo.load_model("yolo_v5s_person_det--512x512_quant_n2x_orca_1")

# set model parameters
model.image_backend = 'opencv' # select OpenCV backend: needed to have overlay image in OpenCV format
model.input_numpy_colorspace = 'BGR'
model.overlay_show_probabilities = True
model.overlay_line_width = 1
model._model_parameters.InputImgFmt = ['JPEG']

# video input and output
input_filename = '1.mp4'
orig_path = Path(input_filename)
ann_path = orig_path.with_name(orig_path.stem + "_annotated" + orig_path.suffix)

# AI prediction loop
# Press 'x' or 'q' to stop
with mytools.Display("AI Camera") as display, \
     mytools.open_video_stream(input_filename) as stream, \
     mytools.open_video_writer(str(ann_path), stream.get(cv2.CAP_PROP_FRAME_WIDTH), stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) as writer:

    fps = 15
    tracker = BYTETracker(
        args=dict_dot_notation({
            'track_thresh': 0.3,
            'track_buffer': fps * 2,
            'match_thresh': 0.8,
            'mot20': False,
        }),
        frame_rate=fps
    )
    timeout_count_dict = {}
    is_counted_dict = {}
    trail_dict = {}
    timeout_count_initial = fps
    
    progress = mytools.Progress(int(stream.get(cv2.CAP_PROP_FRAME_COUNT)))
    for batch_result in model.predict_batch(mytools.video_source(stream, report_error=False)):
        # object detection
        results = batch_result.results
        bboxes = np.zeros((len(results), 5))
        image = batch_result.image

        # byte track
        for index, result in enumerate(results):
            bbox = np.array(result.get('bbox', [0, 0, 0, 0]))
            score = result.get('score', 0)
            bbox_and_score = np.append(bbox, score)
            bboxes[index] = bbox_and_score

        online_targets = tracker.update(bboxes, (1, 1), (1, 1))
        online_target_set = set([])

        # tracking start or continue
        for target in online_targets:
            tid = str(target.track_id)
            online_target_set.add(str(tid))

            box = tuple(map(int, target.tlbr)) # x1 y1 x2 y2
            center = tuple(map(int, target.tlwh_to_xyah(target.tlwh)[:2]))
            if trail_dict.get(tid, None) is None:
                trail_dict[tid] = []
            if is_counted_dict.get(tid, None) is None:
                is_counted_dict[tid] = False
            if not is_counted_dict[tid] and len(trail_dict[tid]) > 1:
                trail_start = trail_dict[tid][0]
                trail_end = center
                is_cross = intersect(line_start, line_end, trail_start, trail_end)
                if is_cross:
                    if trail_start[0] > trail_end[0]:
                        left += 1
                    if trail_start[0] < trail_end[0]:
                        right += 1
                    if trail_start[1] < trail_end[1]:
                        top += 1
                    if trail_start[1] > trail_end[1]:
                        bottom += 1
                    is_counted_dict[tid] = True
            trail_dict[tid].append(center)
            timeout_count_dict[tid] = timeout_count_initial
            if len(trail_dict[tid]) > 1:
                cv2.polylines(image, [np.array(trail_dict[tid])], False, (255, 255, 0))
            cv2.rectangle(image, box[0:2], box[2:4], color=(0, 255, 0), thickness=1)
            cv2.drawMarker(image, center, (255, 255, 0), markerType=cv2.MARKER_CROSS)
            cv2.putText(image, tid, box[0:2], cv2.FONT_HERSHEY_PLAIN, 1, color=(0, 255, 0), thickness=1)

        # tracking terminate
        for tid in set(timeout_count_dict.keys()) - online_target_set:
            timeout_count_dict[tid] -= 1
            if timeout_count_dict[tid] == 0:
                del timeout_count_dict[tid], is_counted_dict[tid], trail_dict[tid]

        text = 't:{} b:{} l:{} r:{}'.format(top, bottom, left, right)
        cv2.putText(image, text, (10, image.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1, color=(0, 255, 0), thickness=1)
        cv2.line(image, line_start, line_end, (0, 255, 0))
        
        writer.write(image)
        display.show(image)
        progress.step()