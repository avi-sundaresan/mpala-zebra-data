# Grevy's Zebra Camera Trap ID Dataset

This repository contains the filtered and labeled camera trap data for the paper **"Adapting the re-ID challenge for static sensors"**. Due to the sensitivity of the data, GPS information has been obfuscated. 

## Data Description

The dataset includes filtered annotations from a network of 70 camera traps deployed at the Mpala Research Centre. The data spans two years and consists of:
- **8.9 million raw images** filtered to yield **685 high-quality annotations**.
- Each folder in the dataset represents a unique Grevy's zebra individual, with a total of **173 folders**. 
- Annotations scored for quality (Census Annotation confidence score above 0.31). 
- Encounter definitions based on temporal clustering of images taken within the same minute by a single camera.
