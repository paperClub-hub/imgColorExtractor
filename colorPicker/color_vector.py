#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @ Date: 2022-07-05 10:30
# @ Author: NING MEI

import json
import os
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from scipy.cluster.vq import vq, kmeans
from orderedset import OrderedSet


""" 图像颜色向量转化 """


def load_color_definitions(BASE_PATH):
	# 参考： https://xkcd.com/color/rgb/
	""" 读取颜色配置文件 """

	COLOR_DEF_PATH = get_data(BASE_PATH)
	try:
		with open(COLOR_DEF_PATH) as json_file:
			color_definitions = json.load(json_file)
	except IOError:
		raise IOError(" 未发现配置文件 %s" % COLOR_DEF_PATH)

	# rgb颜色对应表
	color_names = list()
	color_vectors = list()
	for color_name, color_vector_list in color_definitions.items():
		for color_vector in color_vector_list:
			color_names.append(color_name)
			color_vectors.append(color_vector)
	code_book = np.array(color_vectors)

	return code_book, color_names


def color_matcher(code_book, color_names, image, method = 'vq'):

	"""颜色聚类、计算 ：image 为rgb 数组图像。"""
	# TODO: 需要确定算法

	if not isinstance(image, np.ndarray):
		image = np.array(image)
	assert len(image.shape) >= 3 , "输入为彩色图像 ！"

	h, w, _ = image.shape
	image_array = np.reshape(image[:, :, :3], (w * h, 3))
	assert 0.0 <= max(image_array[:,1]) <= 255.0, "RGB 颜色数值超界 ！"
	if max(image_array[:,1]) <= 1.0:
		image_array = image_array * 255


	if method == 'vq':
		labels, dist = vq(image_array, code_book)
		img_labels = np.empty((w * h), dtype=object)

		for idx, label in np.ndenumerate(labels):
			img_labels[idx] = color_names[label]
		img_labels = np.reshape(img_labels, (w, h))
		return img_labels

	elif method == 'KDTree':
		img_labels = np.empty((w * h), dtype=object)
		nestTree = KDTree(code_book)
		dist, labels = nestTree.query(image_array)

		for idx,label in np.ndenumerate(labels):
			img_labels[idx] = color_names[label]

		img_labels = np.reshape(img_labels, (w, h))

		return img_labels





def bool_colors(image, color_def_path=None):
	""" 获取提取 bool 图像 """

	if not isinstance(image, np.ndarray):
		image = np.array(image)

	if color_def_path:
		code_book, color_names = load_color_definitions()
	else:
		code_book, color_names = CODE_BOOK, COLOR_NAMES

	h, w, _ = image.shape
	image_array = np.reshape(image[:, :, :3], (w * h, 3))
	flatten_values = image_array[:, 1]
	assert 0.0 <= max(flatten_values) <= 255.0, "RGB 颜色数值超界 ！"

	if max(flatten_values) <= 1.0:
		image_array = image_array * 255

	labels, _ = vq(image_array, code_book) ### code_book 最近的index 和 距离分值
	img_labels = np.empty((w * h), dtype=object)
	for idx, label in np.ndenumerate(labels):
		img_labels[idx] = color_names[label]
	img_labels = np.reshape(img_labels, (h, w))


	bool_arrays = dict()
	for color_name in set(color_names):
		bool_array = img_labels == color_name
		bool_arrays[color_name] = bool_array

	del img_labels, image, image_array, flatten_values
	return bool_arrays


def rgb_colors(image, color_def_path=None):
	""" 获取提取 rgb图像 """

	if not isinstance(image, np.ndarray):
		image = np.array(image)

	bool_arrays = bool_colors(image, color_def_path)

	rgb_arrays = dict()
	for color_name in bool_arrays:
		rgb_arrays[color_name] = image[:, :, :3].copy()
		bg_color = 255 if color_name == "achro" else 0
		rgb_arrays[color_name][np.invert(bool_arrays[color_name])] = bg_color

	del image, bool_arrays
	return rgb_arrays



def extract(image, color_def_path=None):
	""" 颜色统计 """
	if not isinstance(image, np.ndarray):
		image = np.array(image)

	bool_arrays = bool_colors(image, color_def_path=color_def_path)

	counts = dict()
	rgb_arrays = dict()
	for color_name in bool_arrays:
		counts[color_name] = np.sum(bool_arrays[color_name])
		rgb_arrays[color_name] = image[:, :, :3].copy()
		# bg_color = 255 if color_name == "achro" else 0
		bg_color = 0
		rgb_arrays[color_name][np.invert(bool_arrays[color_name])] = bg_color

	c_names = list(OrderedSet(COLOR_NAMES))
	data = ([counts.get(name)*1.0 / sum(counts.values()) for name in c_names ], [name for name in c_names])

	del image, bool_arrays
	return data, rgb_arrays


BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
def get_data(path=BASE_PATH):
	return path+'/color_definitions.json'

CODE_BOOK, COLOR_NAMES = load_color_definitions(BASE_PATH)

