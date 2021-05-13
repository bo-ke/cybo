# Cybo

Cybo 是一款基于 TensorFlow 2.0 的轻量级自然语言处理（NLP）工具包，目标是快速实现 NLP 任务以及构建复杂模型。

Cybo 提供多种神经网络组件以及复现模型 - 涵盖文本分类、命名实体识别(NER)、语意理解-意图槽位联合模型(SLU)

## 模型支持

### 文本分类

| name                          |Description | demos |
| ----------------------------- |----------- | ----- |
| TextCNN                       |            |       |
| transformer                   |            |       |
| BertForSequenceClassification |            |       |

### NER (命名实体识别)

| name                       | Description | demos |
| -------------------------- | ----------- | ----- |
| BiLSTM+CRF                 |             |       |
| BertForTokenClassification |             |       |

### SLU (意图槽位联合模型)

| name                | Description | tutorials                                                                       |
| ------------------- | ----------- | -----                                                                           |
| SlotGated           |             | [slot_gated](https://github.com/bo-ke/cybo/tree/master/tutorials/slu/slot_gated)|
| SF-ID               |             | [SF-ID](https://github.com/bo-ke/cybo/tree/master/tutorials/slu/sf_id)          |
| StackPropagationSLU |             |                                                                                 |
| BertSLU             |             |                                                                                 |
