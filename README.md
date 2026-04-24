# LLaMA-Factory Loss Weight Extension

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![LLaMA-Factory](https://img.shields.io/badge/LLaMA--Factory-0.9.1-green.svg)](https://github.com/hiyouga/LLaMA-Factory)
[![Transformers](https://img.shields.io/badge/Transformers-4.46.1-orange.svg)](https://github.com/huggingface/transformers)

## 目录

- [项目介绍](#项目介绍)
- [功能特性](#功能特性)
- [文件结构](#文件结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [代码修改详情](#代码修改详情)
- [技术原理](#技术原理)
- [应用场景](#应用场景)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 项目介绍

本项目在 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架基础上实现了**样本级损失加权功能**的扩展，支持在微调过程中灵活配置每个样本的损失权重，从而提升模型对关键样本的关注度。

在传统的模型微调过程中，所有训练样本对损失函数的贡献是均等的。然而，在实际应用场景中，不同样本往往具有不同的重要性。例如，高质量标注数据、关键领域知识样本或特定任务的核心样本，应当获得更高的训练权重。本扩展正是为解决这一需求而设计，通过在数据层面引入权重标识，并在训练链路中实现权重传递与损失加权计算，使模型能够更加精准地学习关键样本的特征。

### 核心价值

- **差异化训练**：通过权重配置实现样本级别的差异化训练
- **质量导向**：提升高质量样本对模型训练的影响力
- **领域适配**：针对特定领域任务强化关键知识的学习

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **样本级权重控制** | 为每个训练样本独立设置损失权重，支持任意正浮点数值 |
| **无缝集成** | 基于 LLaMA-Factory v0.9.1 原生扩展，保持原有功能完整 |
| **灵活配置** | 通过简单的 JSON 配置即可启用加权功能，无需修改训练脚本 |
| **兼容性** | 支持 Label Smoothing，与原有训练参数完全兼容 |

---

## 文件结构

```
llamafactory-loss-weight/
│
├── README.md                           # 项目主文档（本文件）
├── LICENSE                             # Apache 2.0 许可证
├── .gitignore                          # Git 忽略文件配置
│
├── docs/
│   └── TECHNICAL_DETAILS.md            # 技术详解文档
│
└── examples/
    ├── dataset_info.json               # 数据集配置示例
    └── train_data.json                 # 训练数据格式示例
```

### 文件说明

| 文件 | 说明 |
|------|------|
| `README.md` | 项目主文档，包含项目介绍、使用方法、代码修改详情等内容 |
| `LICENSE` | Apache 2.0 开源许可证文件 |
| `.gitignore` | Git 版本控制的忽略文件配置 |
| `docs/TECHNICAL_DETAILS.md` | 技术详解文档，记录完整的代码追踪过程与实现细节 |
| `examples/dataset_info.json` | 数据集配置示例，展示如何在 LLaMA-Factory 中配置 loss_weight 字段 |
| `examples/train_data.json` | 训练数据格式示例，展示如何为样本设置权重值 |

---

## 环境要求

```
llamafactory=0.9.1
transformers=4.46.1
```

> **注意**：本扩展基于 LLaMA-Factory v0.9.1 版本开发，请确保版本一致性以避免兼容性问题。

---

## 快速开始

### Step 1: 代码修改

按照 [代码修改详情](#代码修改详情) 章节，修改 `src/llamafactory` 中的相关代码文件。

### Step 2: 数据集配置

在 `dataset_info.json` 的 `columns` 中增加 `loss_weight` 字段：

```json
"dataset_example": {
    "file_name": "your_data.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images": "images",
        "loss_weight": "loss_weight"
    }
}
```

### Step 3: 数据格式

为每个样本设置 `loss_weight` 取值。若不设置可能会报错，设置为 `1.0` 时与原版行为相同：

```json
{
    "messages": [
        {
            "role": "user",
            "content": "<image>你是一个道路安全检查AI……"
        },
        {
            "role": "assistant",
            "content": "否"
        }
    ],
    "images": [
        "path/to/image.jpg"
    ],
    "loss_weight": 1.0
}
```

---

## 使用方法

### 权重设置建议

| 权重值 | 适用场景 |
|--------|----------|
| `0.5` | 低质量样本或辅助性数据 |
| `1.0` | 普通样本（默认行为） |
| `1.5` | 较高质量样本 |
| `2.0` | 高质量核心样本 |
| `3.0+` | 关键领域知识或特殊任务样本 |

### 训练命令

配置完成后，使用常规的 LLaMA-Factory 训练命令即可：

```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path your_model_path \
    --dataset dataset_example \
    --output_dir output \
    --finetuning_type lora
```

---

## 代码修改详情

本章节详细说明需要修改的代码文件及其修改内容。

### 1. `llamafactory/data/parser.py`

**DatasetAttr 类定义**：增加属性声明

```python
loss_weight: Optional[float] = None
```

**get_dataset_list 函数**：在 `column_names` 列表中添加 `"loss_weight"`

```python
if "columns" in dataset_info[name]:
    column_names = ["system", "tools", "images", "videos", "chosen", "rejected", "kto_tag", "loss_weight"]
    if dataset_attr.formatting == "alpaca":
        column_names.extend(["prompt", "query", "response", "history"])
    else:
        column_names.extend(["messages"])

    for column_name in column_names:
        dataset_attr.set_attr(column_name, dataset_info[name]["columns"])
```

### 2. `llamafactory/data/aligner.py`

**convert_sharegpt 函数**：在 output 字典中增加 `_loss_weight` 字段

```python
"_loss_weight": example[dataset_attr.loss_weight] if dataset_attr.loss_weight else None,
```

### 3. `llamafactory/data/processors/supervised.py`

**preprocess_supervised_dataset 函数**：增加 loss_weight 传递

```python
model_inputs["loss_weight"].append(examples["_loss_weight"][i])
```

**preprocess_packed_supervised_dataset 函数**：在适当位置增加以下语句

```python
batch_loss_weight = []
```

```python
batch_loss_weight.append(examples["_loss_weight"][i] or [])
```

```python
packed_loss_weight = []
```

```python
packed_loss_weight += batch_loss_weight[index]
```

```python
model_inputs["loss_weight"].append(packed_loss_weight or None)
```

### 4. `llamafactory/train/sft/trainer.py`

#### 4.1 导入模块

```python
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from torch import nn
```

#### 4.2 重写 compute_loss 方法

```python
@override
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """

    if (self.label_smoother_weighted is not None or self.compute_loss_func is not None) and "labels" in inputs:
        # 将self.label_smoother改为self.label_smoother_weighted
        labels = inputs.pop("labels")
    else:
        labels = None

    if "loss_weight" in inputs:
        loss_weight = inputs.pop("loss_weight") # 按照预期，loss_weight是形状为(batchsize,)的torch.tensor
    else:
        loss_weight = None

    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    outputs = model(**inputs, labels=labels)

    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]
    
    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif (model_name == "HikLMMVForCausalLM") or (model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()):
            loss = self.label_smoother_weighted(outputs, labels, loss_weight, shift_labels=True)
        else:
            loss = self.label_smoother_weighted(outputs, labels, loss_weight)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    
    r"""
    Fixes the loss value for transformers 4.46.0.
    https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
    """

    if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
        loss = loss / self.args.gradient_accumulation_steps
        
    return (loss, outputs) if return_outputs else loss
```

#### 4.3 新增 label_smoother_weighted 方法

```python
def label_smoother_weighted(self, model_output, labels, loss_weight, shift_labels=False):

    # self.label_smoother在初始化时使用ignore_index默认值-100
    ignore_index = -100
    epsilon=self.args.label_smoothing_factor
    
    logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    log_probs = -nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
                                                                                                            
    nll_loss = log_probs.gather(dim=-1, index=labels)
    
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    if loss_weight is not None:
        weights = loss_weight.unsqueeze(-1).unsqueeze(-1) # 将weights的形状变为(batchsize,1,1)
        nll_loss = nll_loss * weights # 损失函数乘权重
        smoothed_loss = smoothed_loss * weights

    else:
        raise ValueError("错误：loss_weight is None!")

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss
```

#### 4.4 重写 _set_signature_columns_if_needed 方法

保留 `loss_weight` 字段信息：

```python
@override
def _set_signature_columns_if_needed(self):
    super()._set_signature_columns_if_needed()
    self._signature_columns += ["loss_weight"]
```

---

## 技术原理

### 数据流传递链路

```
数据集 JSON 文件
       ↓
DatasetAttr（parser.py）解析 loss_weight 字段
       ↓
convert_sharegpt（aligner.py）转换为 _loss_weight
       ↓
preprocess_supervised_dataset（supervised.py）传递至 model_inputs
       ↓
CustomSeq2SeqTrainer._set_signature_columns_if_needed 保留字段
       ↓
compute_loss 获取 loss_weight 并传递
       ↓
label_smoother_weighted 实现加权损失计算
```

### 损失加权公式

加权损失计算的核心公式如下：

$$L_{weighted} = w \cdot L_{nll}$$

其中：
- $w$ 为样本权重（loss_weight）
- $L_{nll}$ 为原始负对数似然损失

当启用 Label Smoothing 时：

$$L_{final} = (1 - \epsilon) \cdot L_{weighted} + \epsilon \cdot L_{smoothed}$$

其中 $\epsilon$ 为 label smoothing 因子。

---

## 应用场景

### 场景一：质量差异化训练

在数据集质量参差不齐的情况下，为高质量标注样本设置更高的权重，使模型更加关注可靠数据。

### 场景二：领域知识强化

针对特定领域的核心知识样本设置较高权重，强化模型在该领域的表现能力。

### 场景三：任务优先级控制

在多任务学习场景中，通过权重配置控制不同任务样本的训练优先级。

### 场景四：样本难度调节

对于困难样本或边界案例，可通过提高权重来增加模型的学习强度。

---

## 致谢

本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 开发，感谢原作者的优秀工作。

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。
