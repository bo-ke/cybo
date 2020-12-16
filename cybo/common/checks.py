# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: checks.py
@time: 2020/12/16 22:26:52

这一行开始写关于本文件的说明与解释


'''


class ConfigurationError(Exception):
    """
    The exception raised by object when it's mis configured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)
