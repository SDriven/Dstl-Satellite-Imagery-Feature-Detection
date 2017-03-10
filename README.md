# Dstl-Satellite-Imagery-Feature-Detection
Place 18 (Public LB) solution for the [Dstl feature detection kaggle challenge](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection) from [DeepVoltaire](https://www.kaggle.com/voltaire) and [Hao](https://www.kaggle.com/lihaorocky) 

The goal was to find ten potentially overlapping features (buildings, other structures, roads, tracks, trees,
crops, rivers, lakes, trucks, cars) in satellite images. This solution uses the [U-Net](https://arxiv.org/abs/1505.04597) neural network
architecture to segment the images for ten binary classes.

## Example input image
![Raw training image](Raw-6120_2_2.png?raw=true "One training image in RGB")

## Example output feature detection
![Binary segmentations](Image-6120_2_2.png?raw=true "Ten binary segmentations for all classes")

#### Thanks to visoft, n01z3, Sergey Mushinskiy, Konstantin Lopuhin for the great scripts and discussions.
