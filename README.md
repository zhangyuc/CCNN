Codes for the paper 'Convexified Convolutional Neural Networks' (https://arxiv.org/abs/1609.01000).

To run the code, download the datasets (http://cs.stanford.edu/people/zhangyuc/datasets/ccnn_data.tar.gz), extract it to the /data directory, then execute python CCNN.py or python TFCNN.py in the /src directory.

The CCNN.py file implements the CCNN model based on numpy and numexpr.

The TFCNN.py file implements the baseline CNN model based on Tensorflow.

Note: the current CCNN implementation caches all feature vectors in the memory. It results in a high memory requirement (~10GB for the mnist datasets, ~50GB for the cifar10 dataset).


