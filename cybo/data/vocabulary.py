# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: vocabulary.py
@time: 2020/12/17 23:56:13

这一行开始写关于本文件的说明与解释


'''
from typing import Dict, List
from collections import Iterable, defaultdict

DEFAULT_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self,
                 namespaces: Iterable[str],
                 padded_function: Callable[[], Any]):
        self._namespaces = namespaces
        self._padded_function = padded_function
        super().__init__()

    def __missing__(self, key: str):
        value = self.padded_function()
        dict.__setitem__(self, key, value)
        return value


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, namespaces, padding_token, oov_token):
        super().__init__(namespaces, padded_function=lambda: {
            padding_token: 0, oov_token: 1})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, namespaces, padding_token, oov_token):
        super().__init__(namespaces, padded_function=lambda: {
            0: padding_token, 1: oov_token})


class Vocabulary():
    def __init__(self,
                 counter=None,
                 namespaces: Iterable[str] = DEFAULT_NAMESPACES,
                 padding_token: str = DEFAULT_PADDING_TOKEN,
                 oov_token: str = DEFAULT_OOV_TOKEN):

        self._namespaces = namespaces
        self._padding_token = padding_token
        self._oov_token = oov_token

        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
