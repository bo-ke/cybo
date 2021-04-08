# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: vocabulary.py
@time: 2020/12/17 23:56:13

这一行开始写关于本文件的说明与解释


'''
import os
from typing import Dict, List, Callable, Iterable, Any, Set, Optional, Union
from collections import defaultdict
from tqdm import tqdm

from loguru import logger

DEFAULT_NON_PADDED_NAMESPACES = ("tags", "labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = "non_padded_namespaces.txt"


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
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # 新增 non_padded_namespaces
        self._non_padded_namespaces.update(non_padded_namespaces)


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    """token_to_index dict: {namespace_1: {}, namespace_2: {...}}
    """

    def __init__(self, non_padded_namespaces, padding_token, oov_token):
        super().__init__(
            non_padded_namespaces=non_padded_namespaces,
            padded_function=lambda: {padding_token: 0, oov_token: 1},
            non_padded_function=lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    """index_to_token dict: {namespace_1: {}, namespace_2: {...}}
    """

    def __init__(self, non_padded_namespaces, padding_token, oov_token):
        super().__init__(
            non_padded_namespaces=non_padded_namespaces,
            padded_function=lambda: {0: padding_token, 1: oov_token},
            non_padded_function=lambda: {})


class Vocabulary():
    def __init__(
            self, counter: Dict[str, Dict[str, int]] = None,
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
        self._extend(counter=counter, min_count=min_count,
                     non_padded_namespaces=non_padded_namespaces)

    def _extend(
            self, counter: Dict[str, Dict[str, int]],
            min_count: Dict[str, int] = None,
            non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
            tokens_to_add: Dict[str, List[str]] = None):
        """往vocab中添加元素
        Args:
            counter (Dict[str, Dict[str, int]]): {namespace: {$word: count, ...}, ...}
            min_count (Dict[str, int], optional): 计入vocab的namespace对应token最小count, {namespace: min_count, ...}
            non_padded_namespaces (Iterable[str], optional): non_padded_namespace. Defaults to DEFAULT_NON_PADDED_NAMESPACES.
            tokens_to_add (Dict[str, List[str]], optional): 添加到namespace中的tokens, {namespace: [token, token, ...], ...}.
        """
        min_count = min_count or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}
        # 返回默认namespaces集合 与 参数返回namespace集合
        # current_namespaces 一般情况下为空
        # current_namespaces = {*self._token_to_index}
        # extension_namespaces = {*counter, *tokens_to_add}
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        # self._non_padded_namespaces.update(non_padded_namespaces)
        # 添加counter中元素到counter
        for namespace in counter:
            token_counters = list(counter[namespace].items())
            # 对counter中元素按频次从高到低排序
            token_counters.sort(key=lambda x: x[1], reverse=True)
            for token, count in token_counters:
                if count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)
        # 添加tokens_to_add中元素到vocab
        for namespace, tokens in tokens_to_add:
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    def add_token_to_namespace(self, token, namespace):
        """往namespace dict中添加token
        Args:
            token ([str]): 添加的token
            namespace ([str]): token对应的namespace
        """
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        """获取对应namespace的vocab_size
        Args:
            namespace (str, optional):
        Returns:
            int: vocab_size
        """
        return len(self._token_to_index[namespace])

    def get_token_index(self, token: str, namespace: str = "tokens") -> int:
        """获取token index
        Args:
            token (str): 对应的token
            namespace (str, optional): token对应的namespace. Defaults to "tokens".
        Returns:
            int: 返回的index
        """
        try:
            return self._token_to_index[namespace][token]
        except KeyError:
            return self._token_to_index[namespace][self._oov_token]

    @classmethod
    def from_examples(
            cls, examples, min_count: Dict[str, int] = None,
            non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
            padding_token: str = DEFAULT_PADDING_TOKEN,
            oov_token: str = DEFAULT_OOV_TOKEN):
        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        for example in tqdm(examples, desc="building vocab"):
            example.count_vocab_items(counter=namespace_token_counts)
        return cls(
            counter=namespace_token_counts,
            min_count=min_count,
            non_padded_namespaces=non_padded_namespaces,
            padding_token=padding_token,
            oov_token=oov_token)

    def save_to_files(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logger.warning(
                f"vocabulary serialization directory {directory} is not empty")
        with open(os.path.join(directory, NAMESPACE_PADDING_FILE), "w", encoding="utf-8") as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                print(namespace_str, file=namespace_file)

        for namespace, mapping in self._index_to_token.items():
            with open(os.path.join(directory, namespace + ".txt"), "w", encoding="utf-8") as token_file:
                num_tokens = len(mapping)
                start_index = 1 if mapping[0] == self._padding_token else 0
                for i in range(start_index, num_tokens):
                    print(
                        mapping[i].replace("\n", "@@NEWLINE@@"),
                        file=token_file)

    @classmethod
    def from_files(
            cls, directory: Union[str, os.PathLike],
            padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
            oov_token: Optional[str] = DEFAULT_OOV_TOKEN):
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        with open(os.path.join(directory, NAMESPACE_PADDING_FILE), "r", encoding="utf-8") as namespace_file:
            non_padded_namespaces = [namespace_str.strip()
                                     for namespace_str in namespace_file]
            vocab = cls(non_padded_namespaces=non_padded_namespaces,
                        padding_token=padding_token, oov_token=oov_token)

            for namespace_filename in os.listdir(directory):
                if namespace_filename == NAMESPACE_PADDING_FILE:
                    continue
                if namespace_filename.startswith("."):
                    continue
                namespace = namespace_filename.replace(".txt", "")
                if namespace in non_padded_namespaces:
                    is_paddded = False
                else:
                    is_paddded = True
                filename = os.path.join(directory, namespace_filename)
                vocab.set_from_file(
                    filename, is_padded=is_paddded, oov_token=oov_token,
                    namespace=namespace)
        return vocab

    def set_from_file(
            self, filename: str, is_padded: bool = True,
            oov_token: str = DEFAULT_OOV_TOKEN, namespace: str = "tokens"):
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}

        with open(filename, "r", encoding="utf-8") as input_file:
            for i, line in enumerate(input_file):
                index = i+1 if is_padded else i
                token = line.strip().replace("@@NEWLINE@@", "\n")
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], "OOV token not found!"
