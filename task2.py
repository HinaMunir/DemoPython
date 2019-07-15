# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:53:40 2019

@author: Hina
"""

from collections import deque
queue=deque()
queue.append("123")
print(queue)
queue.appendleft('1211')
queue.extend('hgjg123')
print(queue)