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

### NER (命名实体识别)

| name                       | paper | Description | demos |
| -------------------------- | ----- | ----------- | ----- |
| BiLSTM+CRF                 | Title |             |       |
| BertForTokenClassification |       |             |       |

### SLU (意图槽位联合模型)

| name                | paper | Description | tutorials |
| ------------------- | ----- | ----------- | ----- |
| SlotGated           | [Slot-Gated Modeling for Joint Slot Filling and Intent Prediction](https://www.aclweb.org/anthology/N18-2118.pdf) |             | [slot_gated](https://github.com/bo-ke/cybo/tree/master/tutorials/slu/slot_gated)|
| SF-ID               |  [A Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot Filling](https://arxiv.org/pdf/1907.00390v1.pdf)     |       | [SF-ID](https://github.com/bo-ke/cybo/tree/master/tutorials/slu/sf_id)      |
| StackPropagationSLU |       |             |       |
| BertSLU             |
