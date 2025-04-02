from wbia.detecttools.directory import Directory
from wbia_lca._plugin import get_ggr_stats, LCAActor
import numpy as np
import wbia  # NOQA
import tqdm
import cv2
import requests
import json

url = 'http://127.0.0.1:5000/review/identification/lca/refer/'
actor = LCAActor()

aids = ibs.get_valid_aids()

# Start the actor (pre-compute data)
result = actor.start(dbdir=ibs.dbdir, aids=aids, graph_uuid='11111111-1111-1111-1111-111111111111')
# Review match
response = actor.resume()

# send desired annotations to web interface
aids = list(map(int, aids))  # change from int64 to int
data = {
    'aids': aids,
}

data_ = {}
for key in data:
    data_[key] = json.dumps(data[key])

response = requests.post(url, data=data_)

clustering = actor.db._get_existing_clustering()

np.save('curr_clustering', np.array(clustering))
np.save('aid_paths', np.array(ibs.get_image_uris_original(ibs.get_annot_gids(aids))))