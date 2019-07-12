#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans

def image_to_feature(img):
    gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray, None)
    return np.array(descs1)

def pretrain_desc(sto):
    rdn_cancer_img = np.random.choice(sto.allCancerList , size=10, replace=False)
    rdn_nocancer_img = np.random.choice(sto.allNoCancerList , size=10, replace=False)
    cancer_desc = [image_to_feature(elm.content) for elm in rdn_cancer_img]
    no_cancer_desc = [image_to_feature(elm.content) for elm in rdn_nocancer_img]
    
    train_desc =  np.concatenate((cancer_desc, no_cancer_desc))
    train_desc = np.array([elm for elm in train_desc if elm.shape is not ()])
    print(train_desc.shape)  
    train_desc = np.concatenate((train_desc))
    train_desc = train_desc.astype(np.float32)
    # compute mean and center descriptors

    train_mean = np.mean((train_desc))
    train_desc = train_desc - train_mean
    
    train_cov = np.dot(train_desc.T, train_desc)
    eigvals, eigvecs = np.linalg.eig(train_cov)
    perm = eigvals.argsort()                   # sort by increasing eigenvalue
    pca_transform = eigvecs[:, perm[11:61]]   # eigenvectors for the 50 last eigenvalues
    train_desc = np.dot(train_desc, pca_transform)
    
    kmeans = MiniBatchKMeans(n_clusters=128, random_state=0) 
    kmeans.fit(train_desc)
    
    return kmeans, train_mean, pca_transform
        
def compute_descriptors(sto, train_mean, pca_transform, kmeans):
    image_descriptors = np.zeros((sto.size(), kmeans.n_clusters), dtype=np.float32)
    l2_normalizer = sklearn.preprocessing.Normalizer(norm='l2', copy=True)
    for ii  in range(sto.size()):
        img= sto.allList[ii]
        print("Indexing %s" % (sto.allList[ii].name,))
        # read the descriptors
        desc = image_to_feature(img.content)

        if desc is None or desc.shape == () or desc.shape[0] == 0:
            # let the descriptor be 0 for all values
            # note that this is bad and the element should be dropped from the index
            print("WARNING: zero descriptor for %s" % (sto.allList[ii].name,))
            continue
        
        # convert to float
        desc = desc.astype(np.float32)
        
        # center and apply PCA transform
        desc = desc - train_mean
        desc = np.dot(desc, pca_transform)
        
        # get cluster ids
        clabels = kmeans.predict(desc)# FIXME
        # compute histogram
        
        descr_hist = np.histogram(clabels, np.arange(129))[0]
            
        # l1 norm
        descr_hist = descr_hist / np.sum(descr_hist)  # FIXME
        
        # take the sqrt (Hellinger kernel)
        descr_hist = np.sqrt(descr_hist)  # FIXME
        
        # l2 norm
        descr_hist = l2_normalizer.transform(descr_hist.reshape(1,len(descr_hist)))
        descr_hist = descr_hist.flatten()
#         descr_hist = descr_hist / L2  # FIXME
        
        # update the index
        image_descriptors[ii] = descr_hist
    print("Indexing complete.")
    return image_descriptors
    
def compute_all_descriptor(sto):
    kmeans, train_mean, pca_transform = pretrain_desc(sto)
    descs = compute_descriptors(sto, train_mean, pca_transform,kmeans)
    return descs

def make_train_and_test_list(sto, test_proportion):
    descs = compute_all_descriptor(sto)
    gathered = [(sto.allList[i].content,descs[i],sto.allList[i].is_cancer) for i in range(sto.size())]
    cancer = np.array([elm for elm in gathered if elm[2]])
    nocancer = np.array([elm for elm in gathered if not elm[2]])
    max_equi_size = min(len(cancer),len(nocancer))
    
    idx_cancer = np.random.choice(len(cancer),max_equi_size, replace=False)
    idx_nocancer = np.random.choice(len(nocancer),max_equi_size, replace=False)

    rdn_cancer = cancer[idx_cancer]
    rdn_nocancer = nocancer[idx_nocancer]
    rdn_all_img = np.concatenate((rdn_cancer,rdn_nocancer))
    np.random.shuffle(rdn_all_img)
    
    train_size = int(len(rdn_all_img) * (1 - test_proportion))
    print(train_size)
    all_train = rdn_all_img[:train_size]
    all_test = rdn_all_img[train_size:]
    
    train_image = [elm[0] for elm in all_train]
    train_x = [elm[1] for elm in all_train]
    train_y = [elm[2] for elm in all_train]
    
    test_image = [elm[0] for elm in all_test]
    test_x = [elm[1] for elm in all_test]
    test_y = [elm[2] for elm in all_test]
    
    return train_image, train_x, train_y, test_image, test_x, test_y
    

