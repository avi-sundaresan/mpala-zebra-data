from wbia.detecttools.directory import Directory
from wbia_lca._plugin import get_ggr_stats, LCAActor
import numpy as np
import wbia  # NOQA
import tqdm
import cv2

IMPORT_PATH = '/data/import/'
CA_NMS_THRESH = 0.9

def convert_box(bbox, aid):
    xtl, ytl, w, h = ibs.get_annot_bboxes(aid)
    x0, y0, x1, y1 = bbox
    x0 = int(np.around(x0 * w))
    y0 = int(np.around(y0 * h))
    x1 = int(np.around(x1 * w))
    y1 = int(np.around(y1 * h))
    xtl += x0
    ytl += y0
    w -= x0 + x1
    h -= y0 + y1
    bbox = (xtl, ytl, w, h)
    return bbox

def gradient_magnitude(image_filepath):
    try:
        image = cv2.imread(image_filepath)
        image = image.astype(np.float32)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    except Exception:
        magnitude = [-1.0]

    result = {
        'sum': np.sum(magnitude),
        'mean': np.mean(magnitude),
        'max': np.max(magnitude),
    }
    return result

globals().update(locals())

# import the images on disk into the WBIA database
try:
   gid_list = ibs.import_folder(IMPORT_PATH)
except:
   direct = Directory(IMPORT_PATH, recursive=True, images=True)
   filepaths = list(direct.files())

   gids = ibs.get_valid_gids()
   processed = set(ibs.get_image_uris_original(gids))

   filepaths_ = sorted(list(set(filepaths) - processed))
   chunks = ut.ichunks(filepaths_, 128)

   for filepath_chunk in tqdm.tqdm(chunks):
       try:
           ibs.add_images(filepath_chunk)
       except Exception:
           pass

   gid_list = ibs.get_valid_gids()

# Run the detector for GGR2 (GPU compute recommended)
# This provides bounding boxes + species label (NOTE: grevy's/plains are
# combined under generic "zebra" label)

config = {
    'algo'           : 'lightnet',
    'config_filepath': 'ggr2',
    'weight_filepath': 'ggr2',
    'sensitivity'    : 0.4,
    'nms'            : True,
    'nms_thresh'     : 0.4,
}

results_list = ibs.depc_image.get_property('localizations', gid_list, None, config=config)

# Convert the detection results in to annotations
aids_list = ibs.commit_localization_results(gid_list, results_list)
aid_list = ut.flatten(aids_list)

# Filter the annotations for "zebra" class
species_list = ibs.get_annot_species(aid_list)
flag_list = ['zebra' in species for species in species_list]
aid_list_ = ut.compress(aid_list, flag_list)

# Determine the species of zebra (plains vs. grevys) and viewpoints (e.g. front-right)
# (GPU compute recommended)

config = {
    'labeler_algo': 'densenet',
    'labeler_weight_filepath': 'zebra_v1',
}

result_list = ibs.depc_annot.get_property('labeler', aid_list_, None, config=config)

# Retrieve the results and apply to the annotations
species_list = ut.take_column(result_list, 1)
viewpoint_list = ut.take_column(result_list, 2)

print(ut.repr3(ut.dict_hist(species_list)))
print(ut.repr3(ut.dict_hist(viewpoint_list)))

ibs.set_annot_species(aid_list_, species_list)
ibs.set_annot_viewpoints(aid_list_, viewpoint_list)

# Filter for right-side viewpoints of Grevy's zebra
aid_list = ibs.get_valid_aids()
species_list = ibs.get_annot_species(aid_list)
viewpoint_list = ibs.get_annot_viewpoints(aid_list)
aids = [
    aid
    for aid, species, viewpoint in zip(aid_list, species_list, viewpoint_list)
    if species == 'zebra_grevys' and 'right' in viewpoint
]
filtered_aids = aids[:]
filtered_gids = list(set(ibs.get_annot_gids(filtered_aids)))

# Filter out annotations that aren't Census Annotations
config = {'classifier_algo': 'densenet', 'classifier_weight_filepath': 'canonical_zebra_grevys_v4'}
prediction_list = ibs.depc_annot.get_property('classifier', aids, 'class', config=config)
confidence_list = ibs.depc_annot.get_property('classifier', aids, 'score', config=config)
confidence_list = [
   confidence if prediction == 'positive' else 1.0 - confidence
   for prediction, confidence in zip(prediction_list, confidence_list)
]
flag_list = [confidence >= 0.31 for confidence in confidence_list]

# Keep only the CA right-side Grevy's zebras
ca_aids = ut.compress(aids, flag_list)
conf_list = ut.compress(confidence_list, flag_list)
ca_gids = list(set(ibs.get_annot_gids(ca_aids)))

# Compute the CA Regions (smaller bounding box) for each Census Annotation
config = {
   'canonical_weight_filepath': 'canonical_zebra_grevys_v4',
}
prediction_list = ibs.depc_annot.get_property('canonical', ca_aids, None, config=config)
bbox_list = [convert_box(prediction, aid) for prediction, aid in zip(prediction_list, ca_aids)]

# Create new annotations from the CA Regions
gid_list = ibs.get_annot_gids(ca_aids)
viewpoint_list = ibs.get_annot_viewpoints(ca_aids)
species_list = ['zebra_grevys+_canonical_'] * len(ca_aids)

car_aids = ibs.add_annots(
   gid_list,
   bbox_list,
   species_list=species_list,
)
ibs.set_annot_viewpoints(car_aids, viewpoint_list)
ibs.set_annot_detect_confidence(car_aids, conf_list)

# Delete all annotations that aren't CAs
all_aids = ibs.get_valid_aids()
species = ibs.get_annot_species(all_aids)
flags = [val != 'zebra_grevys+_canonical_' for val in species]
delete_aids = ut.compress(all_aids, flags)
ibs.delete_annots(delete_aids)

# Check all annotations for blurriness (gradient magnitude), eliminate outliers beyond +/-1.5 stddev
chips_paths = ibs.get_annot_chip_fpath(car_aids)
arg_iter = list(zip(chips_paths))
gradient_dict_list = ut.util_parallel.generate2(
   gradient_magnitude, arg_iter, ordered=True
)
gradient_dict_list = list(gradient_dict_list)
gradient_mean_list = ut.take_column(gradient_dict_list, 'mean')
gradient_thresh_mean = np.mean(gradient_mean_list)
gradient_thresh_std = np.std(gradient_mean_list)
gradient_thresh = gradient_thresh_mean - 1.5 * gradient_thresh_std

globals().update(locals())
gradient_flag_list = [
   gradient_mean >= gradient_thresh for gradient_mean in gradient_mean_list
]
gradient_aid_list = ut.compress(car_aids, gradient_flag_list)

# Delete all annotations that didn't pass this quality check
delete_aids = list(set(ibs.get_valid_aids()) - set(gradient_aid_list))
ibs.delete_annots(delete_aids)

# Delete all images that don't have at least one CA-R
gids = ibs.get_valid_gids()
aids = ibs.get_image_aids(gids)
lens = list(map(len, aids))
flags = [length == 0 for length in lens]
delete_gids = ut.compress(gids, flags)
ibs.delete_images(delete_gids, trash_images=False)
