# 定义包的属性
from __future__ import absolute_import

# __all__ = ['element', 'analytic_method', 'kalman_method', 'common_test_parameter', 'common_test_parameter2', ]

from .element import *
from .common_test_parameter2 import *
from .kalman_method import Resolution as Res_k
# from DDFS.kalman_method import Resolution as Res_k





__version__ = '1.0'
__author__ = 'Eason fu'


# 定义初始化行为
def initialize():
    print("Detector fast Design & Test package is initialized.")


# 默认情况下，当导入包时，__init__.py 文件会被执行
# 可以在这里执行任何的初始化代码
initialize()