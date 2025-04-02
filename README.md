# Grevy's Zebra Camera Trap ID Dataset

This repository contains the filtered and labeled camera trap data for the paper **"Adapting the re-ID challenge for static sensors"**. Due to the sensitivity of the data, GPS information has been obfuscated. 

## Data Description

The dataset includes filtered annotations from a network of 70 camera traps deployed at the Mpala Research Centre. The data spans two years and consists of:
- **8.9 million raw images** filtered to yield **685 high-quality annotations**.
- Each folder in the dataset represents a unique Grevy's zebra individual, with a total of **173 folders**. 
- Annotations scored for quality (Census Annotation confidence score above 0.31). 
- Encounter definitions based on temporal clustering of images taken within the same minute by a single camera.

# WBIA Setup for Zebra Detection (GGR)

This repo contains setup instructions and example scripts to run zebra detection and clustering using WBIA with the GGR dataset.

---

## Quickstart

### Set up working directory
```
mkdir -p ~/wbia
cd ~/wbia
```

Copy images into $(pwd)/wbia/import/. 

### Create Docker env file
```
echo """
HOST_UID=$(id -u)
HOST_USER=$(whoami)
""" > $(pwd)/wbia.env
```

### Run container from within ~/wbia
```
docker run \
 -d \
 -p 5000:5000 \
 --gpus '"device=0"' \
 --shm-size=1g \
 --name wbia.ggr \
 -v $(pwd)/db:/data/db \
 -v $(pwd)/cache:/cache \
 -v /PATH/TO/YOUR/DATA:/data/import \
 --env-file $(pwd)/wbia.env \
 --restart unless-stopped \
 wildme/wbia:latest
```

### Logging
```
docker logs --follow wbia.ggr
```

### Initiate IPython embed interactive session inside Docker container
```
<tmux>$ docker exec -it wbia.ggr bash
```

```
<container>$ embed
```

Now you can execute the contents of the scripts in /scripts in the IPython environment. 

