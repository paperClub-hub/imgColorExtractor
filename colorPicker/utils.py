#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @ Date: 2022-07-05 10:10
# @ Author: NING MEI

import math
import numpy as np
from typing import Union, Optional,List, Dict
import matplotlib.pyplot as plt

def colors2image(colors: Union[List, Dict]):
	""" 颜色聚类结果显示 """


	if isinstance(colors, Dict):
		plt.figure(figsize=(12, 9))
		rows_ = math.sqrt(len(colors))
		rows = int(rows_)
		cols = int(rows + 1) if rows_ > rows else rows

		for indx, (name, color_img) in enumerate(colors.items()):
			pimg = color_img
			pimg = pimg.astype(np.uint8)
			plt.subplot(int(rows), int(cols), indx + 1), plt.imshow(pimg)
			plt.title(name), plt.xticks([]), plt.yticks([]), plt.grid(True)

		plt.show()



