#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @ Date: 2022-07-13 14:06
# @ Author: NING MEI


import cv2
import numpy as np


def bfmatcher(cv2_img1, cv2_img2, show_drawMatcher = True):
	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(cv2_img1, None)
	kp2, des2 = orb.detectAndCompute(cv2_img2, None)

	# 暴力匹配 汉明距离匹配特征点
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)

	use_goog = True
	if use_goog:
		threshold = 0.46
		goodMatches = []
		matches = sorted(matches, key=lambda x: x.distance)
		for i in range(len(matches)):
			if (matches[i].distance < threshold * matches[-1].distance):
				goodMatches.append(matches[i])

		result = cv2.drawMatches(cv2_img1, kp1, cv2_img2, kp2, goodMatches, None)


		cv2.namedWindow("orb", cv2.WINDOW_NORMAL)
		cv2.imshow("orb", result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if show_drawMatcher:
		result = cv2.drawMatches(cv2_img1, kp1, cv2_img2, kp2, matches, None)

		cv2.namedWindow("orb", cv2.WINDOW_NORMAL)
		cv2.imshow("orb", result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()



def bfmatcher2(cv2_img1, cv2_img2, show_drawMatcher=True ):
	orb = cv2.ORB_create()

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	kp1, des1 = orb.detectAndCompute(cv2_img1, None)
	kp2, des2 = orb.detectAndCompute(cv2_img2, None)

	# knn match
	matches = bf.knnMatch(des1, des2, k=1)
	# 删除matches里面的空list，并且根据距离排序
	while [] in matches:
		matches.remove([])
	matches = sorted(matches, key=lambda x: x[0].distance)

	if show_drawMatcher:
		# 画出距离最短的前15个点
		result = cv2.drawMatchesKnn(cv2_img1, kp1, cv2_img2, kp2, matches[0:15],
		                            None,
		                            matchColor=(0, 255, 0),
		                           singlePointColor=(255, 0, 255))

		cv2.namedWindow("orb", cv2.WINDOW_NORMAL)
		cv2.imshow("orb", result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def siftmatcher(cv2_img1, cv2_img2, show_drawMatcher=True ):
	""" pip install opencv-python==3.4.2.16
	pip install opencv-contrib-python==3.4.2.16
	(如果contrib安装失败，则用)
	pip install --user opencv-contrib-python==3.4.2.16
	"""
	sift = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = sift.detectAndCompute(cv2_img1, None)
	kp2, des2 = sift.detectAndCompute(cv2_img2, None)

	index_params = dict(algorithm=0, trees=5)

	search_params = dict(checks=20)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# 记录好的点
	goodMatches = [[0, 0] for i in range(len(matches))]

	for i, (m, n) in enumerate(matches):
		if m.distance < 0.7 * n.distance:
			goodMatches[i] = [1, 0]

	draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=goodMatches, flags=0)

	if show_drawMatcher:
		result = cv2.drawMatchesKnn(cv2_img1, kp1, cv2_img2, kp2, matches, None, **draw_params)
		cv2.namedWindow("orb", cv2.WINDOW_NORMAL)
		cv2.imshow("orb", result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


m1 = cv2.imread('./data/1.jpg')
m2 = cv2.imread('./data/2.jpg')
bfmatcher(m1, m2)
# bfmatcher2(m1, m2)
# siftmatcher(m1, m2)
