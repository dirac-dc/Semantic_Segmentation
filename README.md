# Semantic_Segmentation
Adapting a VGG16 network (a deep learning framework that won the Imagenet challenge) into a pixel by pixel predictor to segment salt regions from seismic images as per the **TGS Salt Identification Challenge** posted on Kaggle.

First run the following:

```
cd ./vgg16/
python vgg16.py
```

This should create a saved_model folder to save the weights/architecture of the original VGG16 model. After this refer to the `runfile.ipynb` notebook.
