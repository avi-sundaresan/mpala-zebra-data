from wbia.detecttools.directory import Directory
from wbia_lca._plugin import get_ggr_stats, LCAActor
import numpy as np
import wbia  # NOQA
import tqdm
import cv2

# THE FOLLOWING ARE OPTIONAL QUALITY CHECKS:

# De-duplicate CA Regions based on NMS
aids = ibs.get_valid_aids()
gids = ibs.get_annot_gids(aids)
gid_map = {}
for gid, aid in zip(gids, aids):
   if gid not in gid_map:
       gid_map[gid] = []
   gid_map[gid].append(aid)
​
keep_aids = []
for gid in tqdm.tqdm(gid_map):
   aids = sorted(set(gid_map[gid]))
   keep_aids += ibs.nms_aids(aids, nms_thresh=CA_NMS_THRESH)
​
# Delete the annotations that are duplicates
delete_aids = list(set(all_aids) - set(keep_aids))
ibs.delete_annots(delete_aids)
​
# Get all current annotations
aid_list = ibs.get_valid_aids()
​
# Check all annotations for good aspect ratios, eliminate outliers beyond +/-2.0 stddev
bbox_list = ibs.get_annot_bboxes(aid_list)
aspect_list = [h / w for xtl, ytl, w, h in bbox_list]
aspect_thresh_mean = np.mean(aspect_list)
aspect_thresh_std = np.std(aspect_list)
aspect_thresh_min = aspect_thresh_mean - 2.0 * aspect_thresh_std
aspect_thresh_max = aspect_thresh_mean + 2.0 * aspect_thresh_std
globals().update(locals())
aspect_flag_list = [
   aspect_thresh_min <= aspect and aspect <= aspect_thresh_max
   for aspect in aspect_list
]
aspect_aid_list = ut.compress(aid_list, aspect_flag_list)
​
# Check all annotations for valid width and height (independently), eliminate outliers beyond +/-1.5 stddev
bbox_list = ibs.get_annot_bboxes(aspect_aid_list)
w_list = [w for xtl, ytl, w, h in bbox_list]
h_list = [h for xtl, ytl, w, h in bbox_list]
w_thresh_mean = np.mean(w_list)
w_thresh_std = np.std(w_list)
h_thresh_mean = np.mean(h_list)
h_thresh_std = np.std(h_list)
w_thresh = w_thresh_mean - 1.5 * w_thresh_std
h_thresh = h_thresh_mean - 1.5 * h_thresh_std
globals().update(locals())
w_h_flag_list = [
   w_thresh <= w and h_thresh <= h
   for w, h in zip(w_list, h_list)
]
w_h_aid_list = ut.compress(aspect_aid_list, w_h_flag_list)
​

