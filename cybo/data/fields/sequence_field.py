# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: sequence_field.py
@time: 2020/12/16 23:09:40

这一行开始写关于本文件的说明与解释


'''
from cybo.data.fields.field import Field, DataArray


class SequenceField(Field[DataArray]):
    def sequence_length(self) -> int:
        raise NotImplementedError

    def empty_field(self):
        raise NotImplementedError
