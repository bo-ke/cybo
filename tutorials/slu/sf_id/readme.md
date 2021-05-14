# SF-ID

Paper Address: [https://arxiv.org/pdf/1907.00390v1.pdf](https://arxiv.org/pdf/1907.00390v1.pdf)

## Benchmark

> ATIS Dataset

| model      | Slot (F1) | Intent (Acc) | Sentence (Acc) |
| ---------- | --------- | ------------ | -------------- |
| Official   |   95.75   |    97.76     |    86.79       |
| This repo. |   95.46   |    95.86     |    85.89       |

>> 这里有个小问题，paper源码中是直接拿test_data作为dev数据集，选的最优模型，这个地方会导致最后的结果有所区别，我们改成跟paper源码方式一样
| model      | Slot (F1) | Intent (Acc) | Sentence (Acc) |
| ---------- | --------- | ------------ | -------------- |
| -          |           |              |                |


> Snips Dataset

| model      | Slot (F1) | Intent (Acc) | Sentence (Acc) |
| ---------- | --------- | ------------ | -------------- |
| Official   |   91.43   |    97.43     |     80.57      |
| This repo. |   92.50   |    96.43     |     82.14      |
