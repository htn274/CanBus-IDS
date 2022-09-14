# Semi-supervised Deep Learning Based In-vehicle Intrusion Detection System Using Convolutional Adversarial Autoencoder

This is the implementation of the paper ["Detecting In-vehicle Intrusion via Semi-supervised Learning-based Convolutional Adversarial
Autoencoders"](https://doi.org/10.1016/j.vehcom.2022.100520)

## Requirements & Libraries

- Python3 
- Tensorflow 1.15

## Dataset 

- [Car-Hacking dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset)

## Code

- [preprocessing.py](preprocessing.py): transform csv format into tfrecord format
- [train_test_split.py](train_test_split.py): split total data into train/test/validation set
- [train.py](train.py): train/test the model by using is_train option
- [test_performance.py](test_performance.py): measure the inference time of the model
