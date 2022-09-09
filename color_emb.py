#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @ Date: 2022-07-05 10:30
# @ Author: NING MEI


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from colorPicker import color_domain, color_vector, utils



def get_domain_color(pil_img, numOfcolors = 10):
	""" 提取主要颜色 """

	colors = color_domain.extract(pil_img, numOfcolors)

	print(f"colors: {colors}")


def get_color_emb(pil_img, show_colors=False):
	""" 提取指定颜色 """

	emb_data, image_dict = color_vector.extract(pil_img)
	print(emb_data)
	# 显示颜色提取结果
	if show_colors:
		utils.colors2image(image_dict)

	return emb_data


img_path = 'data/test.png'
img_path = 'data/s000010.jpg'
img_path = 'data/s000069.jpg'
img_path = 'data/color_plate.png'
img_path = 'data/p000173.jpg'
if __name__ == '__main__':
	img = Image.open(img_path)
	get_color_emb(img)
	get_domain_color(img, True)