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

DEFAULT_NON_PADDED_NAMESPACES = ("tags", "labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self,
                 non_padded_namespaces: Iterable[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any]):
        """生成默认vocabulary dict

        Args:
            non_padded_namespaces (Iterable[str]): 不需要做padded的namespaces 如labels、tag 不需要oov与补全标识符
            padded_function (Callable[[], Any]): 生成包含默认key的dict的函数  如lambda: {padding_token: 0, oov_token: 1}
            non_padded_function (Callable[[], Any]): 生成不需要默认key的dict的函数 如lambda: {}
        """
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super().__init__()

    def __missing__(self, key: str):
        """当索引namespace无结果时，返回对应默认值

        Args:
            key (str): 索引namespace

        Returns:
            返回namespace对应的默认dict
        """
        if any(pattern == key for pattern in self._non_padded_namespaces):
            value = self._non_padded_function()
        else:
            value = self.padded_function()
        dict.__setitem__(self, key, value)
        return value


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    """token_to_index dict: {namespace_1: {}, namespace_2: {...}}
    """

    def __init__(self, non_padded_namespaces, padding_token, oov_token):
        super().__init__(non_padded_namespaces=non_padded_namespaces, padded_function=lambda: {
            padding_token: 0, oov_token: 1}, non_padded_function=lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    """index_to_token dict: {namespace_1: {}, namespace_2: {...}}
    """

    def __init__(self, non_padded_namespaces, padding_token, oov_token):
        super().__init__(non_padded_namespaces=non_padded_namespaces, padded_function=lambda: {
            0: padding_token, 1: oov_token}, non_padded_function=lambda: {})


class Vocabulary():
    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
                 padding_token: str = DEFAULT_PADDING_TOKEN,
                 oov_token: str = DEFAULT_OOV_TOKEN):

        self._non_padded_namespaces = non_padded_namespaces
        self._padding_token = padding_token
        self._oov_token = oov_token

        # 初始化token_to_index dict
        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        # 初始化index_to_token dict
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )

        # 往默认token_to_index dict与index_to_token dict中填充数据
        self._extend()

    def _extend(self,
                counter: Dict[str, Dict[str, int]],
                min_count: Dict[str, int] = None,
                non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES):
        min_count = min_count or {}
        non_padded_namespaces = set(non_padded_namespaces)
        
        
