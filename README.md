---
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
metrics:
- f1
- accuracy
model-index:
- name: distilBERT_ESG
  results: []
datasets:
- TrajanovRisto/esg-sentiment
language:
- en
widget:
- text: "Our waste reduction initiatives aim to minimize environmental impact."
---

## Model Link
[distilBERT_ESG](https://huggingface.co/descartes100/distilBERT_ESG)

## Model description

This repository contains a fine-tuned DistilBERT model using the [esg-sentiment dataset](https://huggingface.co/datasets/TrajanovRisto/esg-sentiment). DistilBERT, a distilled version of BERT, is a powerful transformer-based model for natural language processing tasks. The model has been fine-tuned on the ESG (Environmental, Social, and Governance) sentiment dataset, allowing it to capture nuanced sentiments related to sustainability and corporate responsibility.

### Features

- DistilBERT-based architecture
- Fine-tuned on the esg-sentiment dataset
- Optimized for sentiment analysis in the context of ESG

## Usage
```
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained('descartes100/distilBERT_ESG')
model = DistilBertForSequenceClassification.from_pretrained('descartes100/distilBERT_ESG')

text = "Our waste reduction initiatives aim to minimize environmental impact. From recycling programs to waste reduction technologies, we're dedicated to responsibly managing resources."
encoding = tokenizer(text, return_tensors="pt")
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
outputs = model(**encoding)
logits = outputs.logits
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1

for idx, label in enumerate(predictions):
  print(id2label[idx], ':', label)
```
### Result
```
Environmental Negative : 0.0
Environmental Neutral : 0.0
Environmental Positive : 1.0
Social Negative : 0.0
Social Neutral : 0.0
Social Positive : 1.0
Governance Negative : 0.0
Governance Neutral : 0.0
Governance Positive : 0.0
```

## Intended uses & limitations

### Intended Uses

The fine-tuned DistilBERT model is designed for sentiment analysis tasks related to ESG considerations. It can be used to analyze and classify text data, providing insights into the sentiment towards environmental, social, and governance practices.

### Limitations

- The model's performance is directly influenced by the quality and diversity of the training data.
- It may not generalize well to domains outside the ESG context.
- Users are encouraged to validate results on their specific use cases and datasets.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1     | Roc Auc | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:------:|:-------:|:--------:|
| No log        | 1.0   | 77   | 0.3731          | 0.4    | 0.6322  | 0.2647   |
| No log        | 2.0   | 154  | 0.3158          | 0.2342 | 0.5651  | 0.1324   |
| No log        | 3.0   | 231  | 0.2773          | 0.5    | 0.6791  | 0.3382   |
| No log        | 4.0   | 308  | 0.2636          | 0.6049 | 0.7442  | 0.3382   |
| No log        | 5.0   | 385  | 0.2591          | 0.6296 | 0.7569  | 0.3824   |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.1.0+cu118
- Datasets 2.15.0
- Tokenizers 0.15.0