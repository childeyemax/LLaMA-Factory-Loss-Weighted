# LLaMA-Factory 损失加权功能技术详解

## 文档导读

本文档记录了在 LLaMA-Factory 框架中实现**样本级损失加权**功能的完整技术过程。为便于学习和检索，全文按照**数据流顺序**组织——从一条训练数据在 JSON 文件中的原始形态出发，依次经过数据加载、格式对齐、Tokenization 预处理、DataLoader 整理、列过滤，最终到达模型前向传播与损失计算，完整追踪 `loss_weight` 字段在每一阶段的传递与变换。

**阅读建议：**

- **快速上手**：如果你只想知道需要修改哪些代码，请直接跳转到[附录 A：关键代码修改清单](#附录-a关键代码修改清单)，其中包含全部 6 个修改点的索引与简要说明。
- **理解原理**：建议根据 [数据流函数层级图](#数据流函数层级图) 查阅对应章节进行阅读。每一章开头都有"阶段概要"，用 2-3 句话说明该阶段的输入、操作和输出，帮助建立整体认知后再深入细节。
- **查阅特定环节**：利用文末的[附录 B：修改点映射表](#附录-b修改点映射表)，可以快速定位某个修改点在文档中的位置。

---

## 数据流函数层级图

**数据流方向**：从上至下

```
run_sft
├── get_dataset
│   ├── _get_merged_dataset
│   │   ├── get_dataset_list -> List[Dataset]
│   │   ├── load_single_dataset
│   │   └── merge_dataset
│   │       └── convert_sharegpt
│   ├── _get_preprocess_dataset
│   │   └── preprocess_supervised_dataset / preprocess_packed_supervised_dataset
│   │       └── _encode_supervised_example
│   └── split_dataset
├── trainer=CustomSeq2SeqTrainer()
└── trainer.train()
    └── _inner_training_loop
        ├── get_train_dataloader
        │   ├── _remove_unused_columns
        │   │   ├── _set_signature_columns_if_needed
        │   │   └── remove_columns
        │   └── DataLoader
        └── training_step
            ├── _prepare_inputs
            │    └── _prepare_input
            └── compute_loss
                └── LabelSmoother
```

**数据流总览**：整个训练流程从 `run_sft` 入口函数开始，分为三大阶段：

1. **数据准备阶段**（`get_dataset`）：从磁盘加载原始 JSON 数据集，经过格式对齐（`convert_sharegpt`）将用户定义的 `loss_weight` 字段提取为内部字段 `_loss_weight`，再通过预处理函数（`preprocess_supervised_dataset` / `preprocess_packed_supervised_dataset`）完成 Tokenization 并将 `_loss_weight` 转化为模型输入中的 `loss_weight` 列。
2. **训练器初始化阶段**（`trainer = CustomSeq2SeqTrainer()`）：创建训练器实例，通过重写 `_set_signature_columns_if_needed` 方法确保 `loss_weight` 列不会被 DataLoader 的列过滤机制移除。
3. **训练执行阶段**（`trainer.train()`）：在 `_inner_training_loop` 中，DataLoader 构建时执行列过滤（`_remove_unused_columns`），随后每个训练步骤（`training_step`）将数据转移到设备上（`_prepare_inputs`），最终在 `compute_loss` 中通过 `label_smoother_weighted` 方法实现样本级损失加权计算。

---

## 环境要求

```
llamafactory=0.9.1
transformers=4.46.1
```

---

## 一、run_sft — 训练入口函数

> **阶段概要**：`run_sft` 是 SFT 微调的顶层入口函数。它负责依次完成分词器加载、模板获取、数据集准备、训练器初始化和训练执行。`loss_weight` 字段的数据流在此函数中被启动——通过调用 `get_dataset` 进入数据准备管线。

**源码位置**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/workflow.py#L36>

```python
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    data_collator = get_collator(template, data_args, training_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args)
    metric_module = get_metric_module(data_args)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
```

`run_sft` 的核心流程包含三个步骤：

1. **加载并预处理数据集**：`dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)` — 这是 `loss_weight` 进入系统的起点。
2. **训练配置**：初始化 `CustomSeq2SeqTrainer`，将 `dataset_module`（包含处理后的数据集）传入训练器。
3. **训练执行**：`train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)` — 在训练循环中完成损失加权计算。

以下各章将按照数据流方向，依次深入每个环节。

---

## 二、get_dataset — 数据集加载与预处理

> **阶段概要**：`get_dataset` 函数是数据准备管线的总调度器。它依次调用 `_get_merged_dataset`（加载、对齐、融合原始数据集）和 `_get_preprocess_dataset`（Tokenization 预处理），最终通过 `split_dataset` 划分训练/验证集。`loss_weight` 字段在此阶段完成从原始 JSON 字段到模型输入列的完整转化。

### 2.1 get_dataset 函数总览

**源码位置**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/loader.py#L40>

`get_dataset` 函数内部依次执行三个子步骤：

```python
def get_dataset(...):
    dataset = _get_merged_dataset(template, dataset_args, training_args, ...)
    dataset = _get_preprocess_dataset(template, dataset_args, training_args, dataset, ...)
    dataset = split_dataset(dataset, training_args, ...)
    return dataset
```

- **`_get_merged_dataset`**：负责从磁盘加载数据集，完成格式对齐（将 ShareGPT 格式统一为内部标准格式），并合并多个数据集。`loss_weight` 在此阶段从原始 JSON 字段被提取为内部字段 `_loss_weight`。
- **`_get_preprocess_dataset`**：负责对已对齐的数据集进行 Tokenization 预处理，生成模型可直接消费的 `input_ids`、`attention_mask`、`labels` 等张量，同时将 `_loss_weight` 转化为 `loss_weight` 列。
- **`split_dataset`**：根据配置划分训练集和验证集。此阶段不涉及 `loss_weight` 的修改。

### 2.2 _get_merged_dataset — 加载、对齐、融合数据集

> **阶段概要**：`_get_merged_dataset` 负责将用户配置的数据集从磁盘加载到内存，完成格式对齐（ShareGPT → 内部标准格式），并将多个数据集合并为一个统一的 Dataset 对象。`loss_weight` 在此阶段通过 `get_dataset_list` 注册到 `DatasetAttr` 中，并通过 `convert_sharegpt` 从原始数据中提取为 `_loss_weight` 字段。

#### 2.2.1 get_dataset_list — 建立 DatasetAttr 实例列表

**源码位置**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/parser.py>

`get_dataset_list` 函数解析 `dataset_info.json` 配置文件，为每个命名的数据集创建一个 `DatasetAttr` 实例。`DatasetAttr` 是一个数据类（dataclass），用于存储单个数据集的所有配置属性，包括文件路径、格式类型、列名映射等。

**DatasetAttr 类定义**（部分）：

```python
@dataclass
class DatasetAttr:
    ...
    loss_weight: Optional[float] = None
```

`get_dataset_list` 函数在解析 `dataset_info.json` 中的 `columns` 配置时，会将用户指定的列名映射到 `DatasetAttr` 的对应属性上。具体而言，`column_names` 列表中预先定义了所有可识别的列名，其中包含 `loss_weight`：

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

当用户在 `dataset_info.json` 中配置了 `loss_weight` 列名后，`dataset_attr.loss_weight` 将被设置为用户指定的列名（例如 `"loss_weight"`），后续流程即可通过该属性名从原始数据中提取权重值。

#### 修改点 1：在 DatasetAttr 中注册 loss_weight 字段

**修改文件**：`src/llamafactory/data/parser.py`

**修改内容**：

1. 在 `DatasetAttr` 类定义中增加属性声明：

```python
loss_weight: Optional[float] = None
```

2. 在 `get_dataset_list` 函数的 `column_names` 列表中添加 `"loss_weight"`：

```python
column_names = ["system", "tools", "images", "videos", "chosen", "rejected", "kto_tag", "loss_weight"]
```

**设计理由**：`DatasetAttr` 是数据集配置的核心数据结构。增加 `loss_weight` 属性使得系统能够识别用户在 `dataset_info.json` 中配置的权重列名，为后续从原始数据中提取权重值提供基础。如果不进行此修改，即使训练数据中包含 `loss_weight` 字段，系统也无法感知其存在。

#### 2.2.2 _load_single_dataset — 加载与对齐单个数据集

`_load_single_dataset` 函数负责加载单个数据集文件，并调用 `align_dataset` 函数将其转换为 LLaMA-Factory 的内部标准格式。

**align_dataset 函数**根据数据集的格式类型（ShareGPT、Alpaca 等）调用对应的转换函数。对于 ShareGPT 格式的数据集，调用 `convert_sharegpt` 函数：

```python
def align_dataset(dataset, dataset_attr, ...):
    if dataset_attr.formatting == "sharegpt":
        dataset = convert_sharegpt(dataset, dataset_attr, ...)
    ...
```

**convert_sharegpt 函数**将 ShareGPT 格式的对话数据转换为统一的内部格式。转换后的每条数据包含 `instruction`、`input`、`output` 等标准字段，以及我们新增的 `_loss_weight` 字段。

#### 修改点 2：在 convert_sharegpt 中输出 _loss_weight

**修改文件**：`src/llamafactory/data/aligner.py`

**修改内容**：在 `convert_sharegpt` 函数的输出字典中增加 `_loss_weight` 字段：

```python
"_loss_weight": example[dataset_attr.loss_weight] if dataset_attr.loss_weight else None,
```

**设计理由**：`convert_sharegpt` 是将原始数据转换为内部标准格式的关键环节。在此处提取 `loss_weight` 并以 `_loss_weight`（带下划线前缀表示内部字段）的形式输出，使得权重值能够随数据集一起流转到后续的预处理阶段。使用条件表达式 `if dataset_attr.loss_weight else None` 确保在用户未配置 `loss_weight` 列时不会报错。

**注意事项**：如果将上述代码中的 `None` 替换为 `1`（即默认权重为 1），可能导致错误难以发觉——因为权重为 1 时与不使用加权的行为相同，用户可能误以为加权功能已生效而实际上并未正确配置。

#### 2.2.3 merge_dataset — 融合数据集

`merge_dataset` 函数将多个已对齐的数据集合并为一个统一的 Dataset 对象。此阶段不涉及 `loss_weight` 的修改，`_loss_weight` 字段随数据集自然传递。

### 2.3 _get_preprocess_dataset — 预处理数据集

> **阶段概要**：`_get_preprocess_dataset` 负责对已对齐的数据集进行 Tokenization 预处理。它根据配置选择非打包模式（`preprocess_supervised_dataset`）或打包模式（`preprocess_packed_supervised_dataset`）的预处理函数，将文本转换为模型可直接消费的张量序列。`loss_weight` 在此阶段从 `_loss_weight` 字段转化为模型输入中的 `loss_weight` 列。

#### 2.3.1 get_preprocess_and_print_func — 获取预处理函数

`_get_preprocess_dataset` 内部调用 `get_preprocess_and_print_func` 来获取对应的预处理函数。根据训练配置（是否启用打包模式），该函数返回 `preprocess_supervised_dataset` 或 `preprocess_packed_supervised_dataset`。

#### 2.3.2 preprocess_supervised_dataset — 非打包模式预处理

> **阶段概要**：在非打包模式下，每个样本独立进行 Tokenization，生成独立的 `input_ids`、`attention_mask`、`labels` 序列。`loss_weight` 在此阶段从 `_loss_weight` 字段直接提取并追加到 `model_inputs` 字典中。

**源码位置**：`src/llamafactory/data/processors/supervised.py`

`preprocess_supervised_dataset` 函数遍历数据集中的每个样本，调用 `_encode_supervised_example` 进行 Tokenization 编码，然后将编码结果收集到 `model_inputs` 字典中：

```python
def preprocess_supervised_dataset(examples, ...):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["prompt"])):
        ...
        encoded = _encode_supervised_example(...)
        model_inputs["input_ids"].append(encoded["input_ids"])
        model_inputs["attention_mask"].append(encoded["attention_mask"])
        model_inputs["labels"].append(encoded["labels"])
    return model_inputs
```

**_encode_supervised_example** 函数负责将单条文本数据（prompt + response）转换为 Token ID 序列，生成 `input_ids`、`attention_mask` 和 `labels`。此函数不涉及 `loss_weight` 的处理。

#### 修改点 3：在非打包模式中追加 loss_weight

**修改文件**：`src/llamafactory/data/processors/supervised.py`

**修改内容**：在 `preprocess_supervised_dataset` 函数的遍历循环中，增加对 `_loss_weight` 的提取：

```python
model_inputs["loss_weight"].append(examples["_loss_weight"][i])
```

**设计理由**：在非打包模式下，每个样本独立处理，因此 `loss_weight` 可以直接从 `_loss_weight` 字段中逐条提取并追加到 `model_inputs` 字典中。提取后的 `loss_weight` 将作为数据集的一个独立列，与 `input_ids`、`attention_mask`、`labels` 并列存在。

#### 2.3.3 preprocess_packed_supervised_dataset — 打包模式预处理

> **阶段概要**：在打包模式下，多个样本被拼接为一个长序列以充分利用 GPU 计算资源。`loss_weight` 在此阶段需要按照打包顺序进行拼接，确保每个 token 位置都能对应到其所属样本的权重值。

**源码位置**：`src/llamafactory/data/processors/supervised.py`

`preprocess_packed_supervised_dataset` 函数的工作流程比非打包模式更复杂：

1. 首先遍历所有样本，调用 `_encode_supervised_example` 对每个样本进行 Tokenization 编码。
2. 然后将编码后的多个样本按照最大序列长度（cutoff length）进行打包（packing），将多个短序列拼接为一个长序列。
3. 打包后的 `input_ids`、`attention_mask`、`labels` 都是拼接后的长张量。

#### 修改点 4：在打包模式中处理 loss_weight

**修改文件**：`src/llamafactory/data/processors/supervised.py`

**修改内容**：对 `preprocess_packed_supervised_dataset` 函数，需要在以下位置分别增加代码：

**步骤 1**：在循环开始前初始化列表：

```python
batch_loss_weight = []
```

**步骤 2**：在遍历样本的循环中收集每个样本的 loss_weight：

```python
batch_loss_weight.append(examples["_loss_weight"][i] or [])
```

**步骤 3**：在打包循环中初始化 packed 列表：

```python
packed_loss_weight = []
```

**步骤 4**：在打包循环中拼接 loss_weight：

```python
packed_loss_weight += batch_loss_weight[index]
```

**步骤 5**：在打包循环结束后追加到 model_inputs：

```python
model_inputs["loss_weight"].append(packed_loss_weight or None)
```

**设计理由**：在打包模式下，多个样本的 token 序列被拼接为一个长序列。相应地，`loss_weight` 也需要按照打包顺序进行拼接。由于打包是按 token 级别进行的，每个样本的 `loss_weight` 值需要扩展为与该样本 token 数量相同的列表，然后与其他样本的权重列表拼接。最终，`loss_weight` 列表中的每个元素对应打包后序列中一个 token 位置所属样本的权重值。

### 2.4 split_dataset — 分割数据集

`split_dataset` 函数根据训练配置中的验证集比例或指定的验证集数据集，将合并后的数据集划分为训练集和验证集。此阶段不涉及 `loss_weight` 的修改，`loss_weight` 列随数据集自然传递。

---

## 三、trainer = CustomSeq2SeqTrainer() — 训练器初始化

> **阶段概要**：在 `run_sft` 中，通过 `CustomSeq2SeqTrainer` 构造函数创建训练器实例。训练器初始化过程中，父类 `Trainer` 会执行签名列检测（`_set_signature_columns_if_needed`），确定哪些数据列应保留在 DataLoader 中。由于 `loss_weight` 不在模型的 `forward` 方法签名中，默认会被过滤掉，因此需要通过重写 `_set_signature_columns_if_needed` 方法来保留该列。

### 3.1 Trainer 继承体系

理解列过滤机制需要先了解 Trainer 的继承体系：

- **`Trainer`**（transformers 库）：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L295>
- **`Seq2SeqTrainer`**（transformers 库，继承自 Trainer）：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer_seq2seq.py#L54>
- **`CustomSeq2SeqTrainer`**（LLaMA-Factory，继承自 Seq2SeqTrainer）：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/trainer.py#L46>

```python
class Trainer:
    # transformers 库基类
    ...

class Seq2SeqTrainer(Trainer):
    # transformers 库 Seq2Seq 扩展
    ...

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    # LLaMA-Factory 自定义训练器
    ...
```

### 3.2 Trainer 初始化流程

`Trainer.__init__` 方法在初始化时会调用 `_set_signature_columns_if_needed`，该方法检查模型的 `forward` 方法签名，确定哪些参数列是模型需要的。只有签名中声明的列才会被保留，其余列在 DataLoader 构建时会被过滤掉。

**Trainer.__init__**（关键部分）：

<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L389>

```python
class Trainer:
    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        ...
    ):
        ...
        self._signature_columns = None
        ...
        # 调用签名列检测
        self._set_signature_columns_if_needed()
```

### 3.3 _set_signature_columns_if_needed — 签名列检测机制

**源码位置**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L938>

```python
def _set_signature_columns_if_needed(self):
    if self._signature_columns is None:
        # 获取模型 forward 方法的参数签名
        model_signature = inspect.signature(self.model.forward)
        self._signature_columns = set(model_signature.parameters.keys())
        # 确保标签列被保留
        if "labels" in self.train_dataset.column_names:
            self._signature_columns.add("labels")
        # 确保标签列被保留
        if "label" in self.train_dataset.column_names:
            self._signature_columns.add("label")
```

该方法通过 Python 的 `inspect` 模块获取模型 `forward` 方法的参数签名，将这些参数名作为需要保留的列名集合。由于 `loss_weight` 不是模型 `forward` 方法的参数，它不在 `_signature_columns` 中，因此在后续的 `_remove_unused_columns` 步骤中会被过滤掉。

#### 修改点 5：在 CustomSeq2SeqTrainer 中保留 loss_weight 列

**修改文件**：`src/llamafactory/train/sft/trainer.py`

**修改内容**：在 `CustomSeq2SeqTrainer` 中重写 `_set_signature_columns_if_needed` 方法：

```python
@override
def _set_signature_columns_if_needed(self):
    super()._set_signature_columns_if_needed()
    self._signature_columns += ["loss_weight"]
```

**设计理由**：通过调用父类的 `_set_signature_columns_if_needed` 方法先获取模型签名列，然后将 `loss_weight` 添加到签名列集合中。这样在后续 `_remove_unused_columns` 执行时，`loss_weight` 列将被视为"需要的列"而不会被过滤掉。这是 `loss_weight` 能够从数据集传递到 `compute_loss` 的关键保障——如果不进行此修改，`loss_weight` 会在 DataLoader 构建阶段丢失。

---

## 四、trainer.train() — 训练执行

> **阶段概要**：`trainer.train()` 启动训练循环。在 `_inner_training_loop` 中，系统首先通过 `get_train_dataloader` 构建 DataLoader（其中包含列过滤步骤），然后在每个 `training_step` 中完成设备转移（`_prepare_inputs`）和损失计算（`compute_loss`）。`loss_weight` 在此阶段最终被用于加权损失计算。

### 4.1 Trainer.train 方法

**源码位置**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L1923>

`Trainer.train` 方法是训练的入口，它负责设置训练状态、调用 `_inner_training_loop` 执行实际训练循环，并返回训练结果。`loss_weight` 的处理逻辑集中在 `_inner_training_loop` 中。

### 4.2 _inner_training_loop — 内层训练循环

`_inner_training_loop` 是训练的核心循环，负责迭代执行训练步骤。在此循环中，与 `loss_weight` 相关的两个关键环节是：

1. **`get_train_dataloader`**：构建训练数据的 DataLoader，其中包含列过滤步骤。
2. **`training_step`**：执行单个训练步骤，其中包含设备转移和损失计算。

### 4.3 get_train_dataloader — 构建 DataLoader

> **阶段概要**：`get_train_dataloader` 负责将预处理后的数据集封装为 PyTorch DataLoader。在构建 DataLoader 之前，会执行 `_remove_unused_columns` 过滤掉模型不需要的数据列。由于我们在修改点 5 中已将 `loss_weight` 添加到签名列集合，此处的列过滤不会移除 `loss_weight`。

#### 4.3.1 _remove_unused_columns — 列过滤机制

**源码位置**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L2995>

```python
def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
    if not self.args.remove_unused_columns:
        return dataset
    self._set_signature_columns_if_needed()
    signature_columns = self._signature_columns
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    ...
    if len(ignored_columns) > 0:
        dataset = dataset.remove_columns(ignored_columns)
    return dataset
```

该方法的逻辑是：首先调用 `_set_signature_columns_if_needed` 确保签名列已设置，然后计算数据集列名与签名列的差集（即"不需要的列"），最后从数据集中移除这些列。

#### 4.3.2 _set_signature_columns_if_needed — 签名列检测（回顾）

如 [3.3 节](#33-_set_signature_columns_if_needed--签名列检测机制) 所述，`_set_signature_columns_if_needed` 在初始化时已被重写，`loss_weight` 已被添加到 `_signature_columns` 中。因此，当 `_remove_unused_columns` 在此处再次调用该方法时，`loss_weight` 不会被列入 `ignored_columns`，从而得以保留。

#### 4.3.3 DataLoader 构建

经过列过滤后的数据集被封装为 PyTorch DataLoader。DataLoader 负责在每个训练迭代中返回一个 batch 的数据，其中包含 `input_ids`、`attention_mask`、`labels` 和 `loss_weight` 等列。

### 4.4 training_step — 训练步骤

> **阶段概要**：`training_step` 是每个训练迭代的核心。它首先通过 `_prepare_inputs` 将数据转移到计算设备（GPU），然后调用 `compute_loss` 计算加权损失。`loss_weight` 在此阶段最终参与损失计算。

#### 4.4.1 _prepare_inputs / _prepare_input — 设备转移

**Trainer._prepare_inputs**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3508>

**Trainer._prepare_input**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3490>

`Trainer.training_step` 调用了 `inputs = self._prepare_inputs(inputs)`，`Trainer._prepare_inputs` 中调用了 `inputs = self._prepare_input(inputs)`。

这两个方法的功能如下：

1. **`Trainer._prepare_input`**：递归地将输入数据中的 `torch.Tensor` 转移到 `self.args.device`（GPU）上。如果启用了 DeepSpeed，还会将浮点数和复数张量的数据类型转换为 DeepSpeed 插件配置的 dtype。对于字典类型的输入，它会递归处理每个值；对于列表/元组类型，也会递归处理每个元素。

```python
def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: self._prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(self._prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": self.args.device}
        if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
        return data.to(**kwargs)
    return data
```

2. **`Trainer._prepare_inputs`**：调用 `_prepare_input` 完成设备转移后，如果启用了 past key-value cache（`self.args.past_index >= 0`），还会将 `self._past` 添加到 inputs 中。

```python
def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    inputs = self._prepare_input(inputs)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
            f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
        )
    if self.args.past_index >= 0 and self._past is not None:
        inputs["mems"] = self._past

    return inputs
```

**关键点**：`_prepare_input` 方法对字典中的所有键值对进行递归处理，因此 `loss_weight` 张量也会被自动转移到 GPU 上，无需额外处理。

#### 4.4.2 compute_loss — 损失计算

> **阶段概要**：`compute_loss` 是损失加权功能的核心实现点。原始的 `Trainer.compute_loss` 使用 `LabelSmoother` 计算标准交叉熵损失（可选 label smoothing）。我们通过重写此方法，将 `loss_weight` 从 inputs 中提取出来，并传递给新增的 `label_smoother_weighted` 方法，实现样本级损失加权。

##### Trainer.compute_loss 原始实现

**源码位置**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3610>

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    outputs = model(**inputs)

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
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss
```

原始实现的关键逻辑：
- 如果配置了 `label_smoother`（即 `label_smoothing_factor != 0`），则从 inputs 中弹出 `labels`，在模型前向传播后使用 `LabelSmoother` 计算损失。
- 对于因果语言模型（Causal LM），使用 `shift_labels=True` 对 logits 和 labels 进行偏移对齐。
- 对于其他模型，直接使用 `LabelSmoother` 计算损失。

##### LabelSmoother 原始实现

**源码位置**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer_pt_utils.py#L544>

```python
@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
```

`LabelSmoother` 的核心逻辑：
1. 从模型输出中提取 logits。
2. 如果是因果语言模型，对 logits 和 labels 进行偏移对齐（`shift_labels=True`）。
3. 计算负对数概率（`log_probs`）。
4. 通过 `gather` 操作获取每个位置的真实标签对应的负对数概率（`nll_loss`）。
5. 计算平滑损失（`smoothed_loss`），即所有类别的负对数概率之和。
6. 使用 `padding_mask` 将填充位置（`ignore_index = -100`）的损失置零。
7. 最终损失为 `(1 - epsilon) * nll_loss + epsilon * smoothed_loss`。

##### CustomSeq2SeqTrainer.compute_loss 原始实现

**源码位置**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/trainer.py#L82>

LLaMA-Factory 原始的 `CustomSeq2SeqTrainer` 在 `Trainer` 的基础上改写了 `compute_loss` 方法，主要用于修复 transformers 4.46.0 版本的损失值问题：

```python
@override
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    """
    Fixes the loss value for transformers 4.46.0.
    https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
    """
    loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
    if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
        # other model should not scale the loss
        if return_outputs:
            return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
        else:
            return loss / self.args.gradient_accumulation_steps

    return loss
```

该原始实现仅在 transformers 4.46.0 版本且模型不支持 `loss_kwargs` 时，将损失值除以梯度累积步数以修正损失值。

##### 修改点 6：重写 compute_loss 实现加权损失

**修改文件**：`src/llamafactory/train/sft/trainer.py`

**前置修改 — 导入模块**：

```python
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from torch import nn
```

**重写 compute_loss 方法**：

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
    outputs = model(**inputs) # 或者为 outputs = model(**inputs, labels=labels)
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
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
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
    
    """
    Fixes the loss value for transformers 4.46.0.
    https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
    """

    if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
        loss = loss / self.args.gradient_accumulation_steps
        
    return (loss, outputs) if return_outputs else loss
```

**与原始实现的关键差异**：

1. **`label_smoother` → `label_smoother_weighted`**：在判断条件中将 `self.label_smoother` 替换为 `self.label_smoother_weighted`，确保使用加权版本的损失计算方法。
2. **提取 `loss_weight`**：新增从 inputs 中弹出 `loss_weight` 的逻辑。按照预期，`loss_weight` 是形状为 `(batch_size,)` 的 `torch.Tensor`。
3. **传递 `loss_weight` 给损失计算方法**：将 `loss_weight` 作为参数传递给 `self.label_smoother_weighted`。
4. **保留 transformers 4.46.0 修复逻辑**：保留了原始 `CustomSeq2SeqTrainer` 中对 transformers 4.46.0 版本损失值的修正逻辑。

**注意事项**：

1. 某些模型的 `forward` 方法可能需要输入 `labels`，此时需要将 `outputs = model(**inputs)` 替换为 `outputs = model(**inputs, labels=labels)`。
2. 由于版本问题，`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`（<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/models/auto/modeling_auto.py#L462>）收录的模型可能不全。根据需要可以将 `elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():` 替换为 `elif (model_name == "模型名") or (model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()):`。

##### label_smoother_weighted — 新增加权损失计算方法

**修改文件**：`src/llamafactory/train/sft/trainer.py`

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

**与原始 LabelSmoother 的关键差异**：

1. **权重乘法**：在计算 `nll_loss` 和 `smoothed_loss` 之后、应用 `padding_mask` 之前，将两者乘以 `loss_weight` 对应的权重。权重通过 `loss_weight.unsqueeze(-1).unsqueeze(-1)` 将形状从 `(batch_size,)` 扩展为 `(batch_size, 1, 1)`，以便与 `(batch_size, seq_len, 1)` 形状的 `nll_loss` 和 `(batch_size, seq_len, 1)` 形状的 `smoothed_loss` 进行广播乘法。
2. **强制校验**：当 `loss_weight` 为 `None` 时，抛出 `ValueError` 异常，确保加权功能不会被静默跳过。
3. **参数来源**：`ignore_index` 直接使用硬编码值 `-100`（与 `LabelSmoother` 默认值一致），`epsilon` 从 `self.args.label_smoothing_factor` 获取。

**加权机制说明**：

`loss_weight` 的形状为 `(batch_size,)`，其中每个元素对应一个样本的权重值。通过 `unsqueeze(-1).unsqueeze(-1)` 扩展为 `(batch_size, 1, 1)` 后，与 `(batch_size, seq_len, 1)` 形状的逐 token 损失进行广播乘法。这意味着同一个样本内的所有 token 位置共享相同的权重值，不同样本的 token 则根据各自的权重进行缩放。最终，加权后的损失仍然除以 `num_active_elements`（非填充 token 的总数），得到加权平均损失。

---

## 五、梯度累积行为

> **阶段概要**：本节说明损失加权在梯度累积场景下的行为。

当启用梯度累积（`gradient_accumulation_steps > 1`）时，`training_step` 会在多个 micro-batch 上累积梯度，然后在累积步数达到阈值时执行一次参数更新。在每个 micro-batch 的 `compute_loss` 中，`loss_weight` 都会独立生效——每个 micro-batch 内的样本按照各自的权重对损失进行加权。

对于 transformers 4.46.0 版本，`compute_loss` 中还包含额外的损失值修正逻辑：当模型不支持 `loss_kwargs` 时，损失值会除以 `gradient_accumulation_steps`。此修正逻辑在加权版本中同样保留。

---

## 六、配置与使用指南

### 6.1 数据集配置

在 `dataset_info.json` 中，需要为使用损失加权功能的数据集配置 `loss_weight` 列名：

```json
{
    "my_dataset": {
        "file_name": "train_data.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "loss_weight": "loss_weight"
        }
    }
}
```

### 6.2 训练数据格式

在训练数据 JSON 文件中，为每个样本设置 `loss_weight` 取值：

```json
[
    {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "你好！有什么可以帮助你的？"}
        ],
        "loss_weight": 2.0
    },
    {
        "conversations": [
            {"from": "human", "value": "介绍一下自己"},
            {"from": "gpt", "value": "我是一个AI助手。"}
        ],
        "loss_weight": 1.0
    }
]
```

**使用说明**：

- 若不设置 `loss_weight`，会报错。设置为 `1.0` 时与原版（不使用加权）行为相同。
- 目前只支持 ShareGPT 格式的训练集。

---

## 附录 A：关键代码修改清单

下表汇总了实现样本损失加权功能所需的全部 6 个代码修改点：

| 编号 | 修改位置 | 修改文件 | 修改说明 |
|------|----------|----------|----------|
| 1 | DatasetAttr + get_dataset_list | `src/llamafactory/data/parser.py` | 在 DatasetAttr 中增加 `loss_weight` 属性，并在列名解析中注册 |
| 2 | convert_sharegpt | `src/llamafactory/data/aligner.py` | 在标准格式输出中增加 `_loss_weight` 字段 |
| 3 | preprocess_supervised_dataset | `src/llamafactory/data/processors/supervised.py` | 在非打包预处理中追加 `loss_weight` 到 model_inputs |
| 4 | preprocess_packed_supervised_dataset | `src/llamafactory/data/processors/supervised.py` | 在打包预处理中拼接并追加 `loss_weight` |
| 5 | _set_signature_columns_if_needed | `src/llamafactory/train/sft/trainer.py` | 重写签名列检测方法，保留 `loss_weight` 列不被过滤 |
| 6 | compute_loss + label_smoother_weighted | `src/llamafactory/train/sft/trainer.py` | 重写损失计算方法，实现样本级加权 |

---

## 附录 B：修改点映射表

下表将每个修改点映射到本文档中的章节位置，方便快速定位：

| 修改点 | 修改说明 | 文档位置 |
|--------|----------|----------|
| 修改点 1 | 在 DatasetAttr 中注册 loss_weight 字段 | [§2.2.1 修改点 1](#修改点-1在-datasetattr-中注册-loss_weight-字段) |
| 修改点 2 | 在 convert_sharegpt 中输出 _loss_weight | [§2.2.2 修改点 2](#修改点-2在-convert_sharegpt-中输出-_loss_weight) |
| 修改点 3 | 在非打包模式中追加 loss_weight | [§2.3.2 修改点 3](#修改点-3在非打包模式中追加-loss_weight) |
| 修改点 4 | 在打包模式中处理 loss_weight | [§2.3.3 修改点 4](#修改点-4在打包模式中处理-loss_weight) |
| 修改点 5 | 在 CustomSeq2SeqTrainer 中保留 loss_weight 列 | [§3.3 修改点 5](#修改点-5在-customseq2seqtrainer-中保留-loss_weight-列) |
| 修改点 6 | 重写 compute_loss 实现加权损失 | [§4.4.2 修改点 6](#修改点-6重写-compute_loss-实现加权损失) |
