# RSNA-MICCAI Brain Tumor Radiogenomic Classification

###### This is using the kaggle RSNA database from the competition. Currently I have used the T1w images only for the predictions.
###### This can be expanded to have something like an ensemble of CNNs for each MRI type like FLAIR, T1w, T1wCE, T2w, which can be combined based on the majority voting logic.
###### Since each image is a slice we can see that first couple (or more) images have very little foreground and mostly covered with black background. These images are not very helpful for making predictions. Therefore we implemented a small logic to determine the image with biggest foreground and select NUM_SAMPLES/2 images on both sides of that slice. NUM_SAMPLES being a hyperparameter can be tweaked. Each MRI type like FLAIR or T2w will have to decide the correct NUM_SAMPLES window.
###### The cluster of image slices are treated as video frames and fed to RESNET-152 like [BATCH_SIZE x NUM_SAMPLES x 64 x 64]. The first layer of RESNET-152 is modified to expect NUM_SAMPLES. Also images are resized before passing to the network.
###### We see some over-fitting with only T1w being considered for predictions. To counter that, I have tried to generalize the training data with SKLEARN KFolds, which showed postive results. With EPOCHS = 100 I still see some overfitting. More experiments can be done by doing data augmentation like random flipping with pytorch library.

###### Libraries - 
  Datascience - Pytorch, Numpy, Sklearn.\
  Data visualization - Seaborn, Matplotlib, Pydicom.\
  Misc - Pandas, Pydicom, tensorboard
