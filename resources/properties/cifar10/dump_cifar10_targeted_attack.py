#!/usr/bin/env python3

# This script generates targeted attack properties for networks trained on the
# CIFAR10 dataset, assuming the model takes 32x32x3 images as input.

import warnings
warnings.filterwarnings("ignore")
from keras.datasets import cifar10
import numpy as np
import argparse

def dumpCIFAR10TargetedAttackProperty(index, epsilon, target):
    (x_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # 이미지를 32x32x3 형태로 유지합니다.
    X = np.array(x_train[index]) / 255.0 # flatten() 제거

    orig = y_train[index][0] 
    if orig == target:
        print("Failed! Original label same as targeted adversarial label!")
        exit()

    # 픽셀 제약 조건을 3차원 형태로 출력합니다.
    # CIFAR10 이미지 크기: 32x32x3
    height, width, channels = X.shape 
    for r in range(height):
        for c in range(width):
            for ch in range(channels):
                # x_{row}_{col}_{channel} 형태로 출력
                pixel_value = X[r, c, ch]
                print(f'x_{r}_{c}_{ch} >= {pixel_value - epsilon}')
                print(f'x_{r}_{c}_{ch} <= {pixel_value + epsilon}')
                
    # 타겟 레이블 제약 조건은 동일합니다.
    for i in range(10): 
        if i != target:
            print('+y{} -y{} <= 0'.format(i, target))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--index', required=True, type=int, help = 'The index of image in the training set')
    p.add_argument('-e', '--epsilon', required=True, type=float, help = 'The perturbation')
    p.add_argument('-t', '--target', required=True, type=int, help = 'The target label')
    opts = p.parse_args()
    return opts

def main():
    opts = parse_args()
    index = opts.index
    epsilon = opts.epsilon
    target = opts.target
    dumpCIFAR10TargetedAttackProperty(index, epsilon, target)

if __name__ == "__main__":
    main()