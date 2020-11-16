
import os
from PIL import Image
import cv2
import numpy as np
import pickle

from sklearn.cluster import MiniBatchKMeans


class KmeansTrain():

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        #self.x = [] #
        #self.descriptors=[]
        #self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform
        self.sift = cv2.SIFT_creat()
        self.dico = []

        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label + '/' + data_type):
                #views = []
                print('item loading')
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    #views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)
                    current_view_path = root + '/' + label + '/' + data_type + '/' + item + '/' + view
                    img= cv2.imread(current_view_path)
                    kp, des= self.sift.detectAndCompute(img, None)
                    # we don't need resize the image or crop.
                    for d in des:
                        self.dico.append(d)
                #self.x.append(views)
                #self.y.append(self.class_to_idx[label])
        k = len(self.classes)*10
        batch_size = len(self.classes)

        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(self.dico)
        kmeans.verbose = False
        filename = 'Kmeans.sav'
        pickle.dump(kmeans, open(filename, 'wb'))

