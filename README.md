# Cybo

Cybo 是一款基于 Tensorflow 2.0 的轻量级自然语言处理（NLP）工具包，目标是快速实现 NLP 任务以及构建复杂模型。

Cybo 提供多种神经网络组件以及复现模型 - 涵盖文本分类、命名实体识别(NER)、语意理解-意图槽位联合模型(SLU)

## 复现模型

### 文本分类

| name                          | paper | Description | demos |
| ----------------------------- | ----- | ----------- | ----- |
| TextCNN                       | Title |             |       |
| transformer                   | Text  |             |       |
| BertForSequenceClassification |       |             |       |

### 命名实体识别

| name                       | paper | Description | demos |
| -------------------------- | ----- | ----------- | ----- |
| BiLSTM+CRF                 | Title |             |       |
| BertForTokenClassification |       |             |       |

### SLU(意图槽位联合模型)

| name                | paper | Description | demos |
| ------------------- | ----- | ----------- | ----- |
| SlotGate            | Title |             |       |
| SF-ID               |       |             |       |
| StackPropagationSLU |       |             |       |
| BertSLU             |
