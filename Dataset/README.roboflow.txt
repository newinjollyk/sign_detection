
signs - v5 2024-05-19 12:56pm
==============================

This dataset was exported via roboflow.com on May 19, 2024 at 5:59 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 8467 images.
Signs are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 2 versions of each source image:
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random Gaussian blur of between 0 and 4.5 pixels
* Salt and pepper noise was applied to 1.64 percent of pixels


