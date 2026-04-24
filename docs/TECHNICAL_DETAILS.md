# LLaMA-Factory 损失加权功能技术详解

本文档详细记录了在 LLaMA-Factory 框架中实现样本级损失加权功能的完整技术过程，包括代码追踪、修改点说明及实现细节。

## 目录

- [环境要求](#环境要求)
- [核心流程概述](#核心流程概述)
- [1. 数据集加载与预处理](#1-数据集加载与预处理)
- [2. 训练配置](#2-训练配置)
- [3. 训练执行](#3-训练执行)

---

## 环境要求

```
llamafactory=0.9.1
transformers=4.46.1
```

## 核心流程概述

LLaMA-Factory 的 SFT 微调流程包含三个核心步骤：

1. **加载并预处理数据集**：`dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)`

2. **训练配置**：初始化 CustomSeq2SeqTrainer
```python
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
```

3. **训练执行**：`train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)`

### 入口函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/workflow.py#L36>

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

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # ……L82：Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        ……
```

**补充说明**：
- `CustomSeq2SeqTrainer` 是 `Trainer` 的子类 (`Seq2SeqTrainer`) 的子类，`self.train` 的定义直接来自于 `Trainer` 类
- `Trainer._inner_training_loop` 方法将 dataset 变为 inputs
- inputs 被用来直接计算 loss

---

## 1. 数据集加载与预处理

`dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)`

### 核心步骤

1. 加载、对齐、融合数据集：`dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)`
2. 预处理数据集：`dataset = _get_preprocessed_dataset(dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False)`
3. 分割数据集：`dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)`

### get_dataset 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/loader.py#L225>

```python
def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """                        

    # 1. 加载、对齐、融合数据集
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    # 2. 预处理数据集
    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )

        # 分割数据集
        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        
        ……
        # 返回一个字典，放入训练集和验证集
        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module
```

---

### 1.1 加载、对齐、融合数据集

`dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)`

#### 核心步骤

1. 建立 DatasetAttr 实例列表：`get_dataset_list(dataset_names, data_args.dataset_dir)`
2. 利用 DatasetAttr 实例中的信息加载、对齐数据集：`datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))`
3. 融合数据集：`merge_dataset(datasets, data_args, seed=training_args.seed)`

#### _get_merged_dataset 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/loader.py#L153>

```python
def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))

    return merge_dataset(datasets, data_args, seed=training_args.seed)
```

---

#### 1.1.1 建立 DatasetAttr 实例列表

`get_dataset_list(dataset_names, data_args.dataset_dir)`

读取数据集文件，返回 DatasetAttr 对象（每个对应一个数据集）构成的列表。

##### get_dataset_list 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/parser.py#L74>

```python
def get_dataset_list(dataset_names: Optional[Sequence[str]], dataset_dir: str) -> List["DatasetAttr"]:
    r"""
    Gets the attributes of the datasets.
    """

    #……L98
    dataset_list: List["DatasetAttr"] = []

    #……L127
    else:
        dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

    dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")
    dataset_attr.set_attr("ranking", dataset_info[name], default=False)
    dataset_attr.set_attr("subset", dataset_info[name])
    dataset_attr.set_attr("split", dataset_info[name], default="train")
    dataset_attr.set_attr("folder", dataset_info[name])
    dataset_attr.set_attr("num_samples", dataset_info[name])

    if "columns" in dataset_info[name]:
        column_names = ["system", "tools", "images", "videos", "chosen", "rejected", "kto_tag"]
        if dataset_attr.formatting == "alpaca":
            column_names.extend(["prompt", "query", "response", "history"])
        else:
            column_names.extend(["messages"])

        for column_name in column_names:
            dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

    if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
        tag_names = (
            "role_tag",
            "content_tag",
            "user_tag",
            "assistant_tag",
            "observation_tag",
            "function_tag",
            "system_tag",
        )
        for tag in tag_names:
            dataset_attr.set_attr(tag, dataset_info[name]["tags"])

    dataset_list.append(dataset_attr)

    return dataset_list
```

##### DatasetAttr 的定义

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/parser.py#L26>

```python
@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """
    # 部分属性
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    system: Optional[str] = None
    images: Optional[str] = None
    messages: Optional[str] = "conversations"
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))
```

---

##### 代码修改1

如果我们给每一个样本增加额外标签 `loss_weight`：

1. 在 dataset_info 的 `"columns"` 中增加 `"loss_weight": "loss_weight"`（与 `"messages"` 平级）
2. 数据集中增加 `"loss_weight": 1`（默认值），注意与 `"messages"` 平级
3. DatasetAttr：初始化时增加语句：`loss_weight: Optional[float] = None`，并在 get_dataset_list 函数中将对应代码替换为：

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

4. 这样 `data_attr.loss_weight = "loss_weight"`，于是 `example[data_attr.loss_weight] = 1`（默认值）

---

#### 1.1.2 利用 DatasetAttr 实例中的信息加载、对齐数据集

`datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))`

##### 主要步骤

1. 加载数据集：`dataset = load_dataset(……)`
2. 对齐数据集：`align_dataset(dataset, dataset_attr, data_args, training_args)`

##### _load_single_dataset 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/loader.py#L45>

```python
def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """

    #……L121
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            trust_remote_code=True,
        )

    #……L150
    return align_dataset(dataset, dataset_attr, data_args, training_args)
```

##### 对齐数据集

`align_dataset(dataset, dataset_attr, data_args, training_args)`

**align_dataset 函数**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/aligner.py#L230>

**convert_sharegpt 函数**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/aligner.py#L137>

```python
def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, data_args=data_args)

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )


def convert_sharegpt(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """

    #……L219
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output
```

---

###### 代码修改2

convert_sharegpt 函数的 output 字典中增加：

```python
"_loss_weight": example[dataset_attr.loss_weight] if dataset_attr.loss_weight else None,
```

**注意**：如果把上面的 None 换成 1，可能导致错误难以发觉！

---

### 1.2 预处理数据集

`dataset = _get_preprocessed_dataset(dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False)`

#### 核心步骤

1. `_get_preprocessed_dataset` 函数利用 `preprocess_func` 函数对 dataset 中的每个样本进行处理
2. 在 sft 阶段，`preprocess_func` 函数是 `preprocess_supervised_dataset` 或者 `preprocess_packed_supervised_dataset` 函数的偏函数

#### _get_preprocessed_dataset 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/loader.py#L176>

#### get_preprocess_and_print_func 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/preprocess.py#L36>

```python
def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """

    #……L192
    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())

    #……L204
    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    #……L222
    return dataset


def get_preprocess_and_print_func(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> Tuple[Callable, Callable]:

    #……L51
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            ……
            preprocess_func = partial(
                preprocess_packed_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

    ……
    return preprocess_func, print_function
```

---

#### preprocess_supervised_dataset 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/processors/supervised.py#L90>

```python
def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs
```

---

##### 代码修改3

对 `preprocess_supervised_dataset` 函数增加语句：

```python
model_inputs["loss_weight"].append(examples["_loss_weight"][i])
```

---

#### preprocess_packed_supervised_dataset 函数

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/data/processors/supervised.py#L130>

```python
def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # TODO: use `position_ids` to achieve packing
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    valid_num = 0
    batch_input_ids, batch_labels, batch_images, batch_videos = [], [], [], []
    lengths = []
    length2indexes = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len - 1,  # reserved for the padding token
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        length = len(input_ids)
        if length > data_args.cutoff_len:
            logger.warning_rank0(f"Dropped lengthy example with length {length} > {data_args.cutoff_len}.")
        else:
            lengths.append(length)
            length2indexes[length].append(valid_num)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_images.append(examples["_images"][i] or [])
            batch_videos.append(examples["_videos"][i] or [])
            valid_num += 1

    model_inputs = defaultdict(list)
    knapsacks = greedy_knapsack(lengths, data_args.cutoff_len - 1)  # reserved for the padding token
    for knapsack in knapsacks:
        packed_input_ids, packed_attention_masks, packed_labels = [], [], []
        packed_images, packed_videos = [], []
        for i, length in enumerate(knapsack):
            index = length2indexes[length].pop()
            packed_input_ids += batch_input_ids[index]
            packed_labels += batch_labels[index]
            packed_images += batch_images[index]
            packed_videos += batch_videos[index]
            if data_args.neat_packing:
                packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
            else:
                packed_attention_masks += [1] * len(batch_input_ids[index])

        if len(packed_input_ids) < data_args.cutoff_len:
            pad_length = data_args.cutoff_len - len(packed_input_ids)
            packed_input_ids += [tokenizer.pad_token_id] * pad_length
            packed_labels += [IGNORE_INDEX] * pad_length
            if data_args.neat_packing:
                packed_attention_masks += [0] * pad_length
            else:
                packed_attention_masks += [1] * pad_length  # more efficient flash_attn

        if len(packed_input_ids) != data_args.cutoff_len:
            raise ValueError("The length of packed example should be identical to the cutoff length.")

        model_inputs["input_ids"].append(packed_input_ids)
        model_inputs["attention_mask"].append(packed_attention_masks)
        model_inputs["labels"].append(packed_labels)
        model_inputs["images"].append(packed_images or None)
        model_inputs["videos"].append(packed_videos or None)

    return model_inputs
```

---

##### 代码修改4

对 `preprocess_packed_supervised_dataset` 函数增加语句：

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

---

## 2. 训练配置

```python
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
```

### Trainer, Seq2SeqTrainer, CustomSeq2SeqTrainer 的初始化

**Trainer**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L295>

**Seq2SeqTrainer**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer_seq2seq.py#L54>

**CustomSeq2SeqTrainer**：<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/trainer.py#L46>

```python
class Trainer:

    #……L389
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
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):

        ……
        #L714
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
        ……


class Seq2SeqTrainer(Trainer):
    @deprecate_kwarg("tokenizer", new_name="processing_class", version="5.0.0", raise_if_both_names=True)
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Union[Dataset, "IterableDataset", "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union["PreTrainedTokenizerBase", "BaseImageProcessor", "FeatureExtractionMixin", "ProcessorMixin"]
        ] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        ……


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        ……

        #L82
        @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
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

### 注意事项

1. `CustomSeq2SeqTrainer` 继承自 `Seq2SeqTrainer`，`Seq2SeqTrainer` 继承自 `Trainer`（后面两个来自 transformers 库）
2. `train` 方法直接来自 `Trainer`
3. 如果 `compute_loss_func=None`，那么 `compute_loss` 方法默认使用 `LabelSmoother` 的 `__call__` 方法计算损失函数
4. `CustomSeq2SeqTrainer` 在 `Trainer` 的基础上改写了 `compute_loss` 方法

---

## 3. 训练执行

`train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)`

### 调用链路

1. `Trainer.train` 方法的返回值是：
```python
return inner_training_loop(
    args=args,
    resume_from_checkpoint=resume_from_checkpoint,
    trial=trial,
    ignore_keys_for_eval=ignore_keys_for_eval,
)
```

2. `inner_training_loop` 为 `self._inner_training_loop` 方法的偏函数

```python
inner_training_loop = find_executable_batch_size(self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size)
```

`find_executable_batch_size` 函数参考 <https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer_utils.py#L764>

3. `self._inner_training_loop` 方法调用 `self.training_step`

```python
tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
```

4. `self.training_step` 调用 `self.compute_loss`

5. `self.compute_loss` 默认通过 `LabelSmoother` 类的 `__call__` 方法计算损失函数

6. `CustomSeq2SeqTrainer` 在 `Trainer` 的基础上改写了 `compute_loss` 方法

---

### 3.1 Trainer.train 方法

训练过程由 `trainer.train` 方法执行，其定义来自于 `Trainer` 类：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L2021>

```python
def train(
    self,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    trial: Union["optuna.Trial", Dict[str, Any]] = None,
    ignore_keys_for_eval: Optional[List[str]] = None,
    **kwargs,
):

    ……
    #L2106：Trainer.train方法返回了inner_training_loop函数的返回值，其定义源于方法self._inner_training_loop
    inner_training_loop = find_executable_batch_size(
        self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
    )
    if args.push_to_hub:
        try:
            # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
            hf_hub_utils.disable_progress_bars()
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            hf_hub_utils.enable_progress_bars()
    else:
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
```

---

### 3.2 Trainer._inner_training_loop 方法

<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L2129>

```python
def _inner_training_loop(
    self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
):

    #L2151
    train_dataloader = self.get_train_dataloader()

    #L2387
    for epoch in range(epochs_trained, num_train_epochs):
        epoch_dataloader = train_dataloader

        #L2415
        epoch_iterator = iter(epoch_dataloader)

        #L2423
        for _ in range(total_updates):
            update_step += 1
            num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
            for inputs in batch_samples:
                ……
                #L2473
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                ……

    #L2638
    return TrainOutput(self.state.global_step, train_loss, metrics)
```

#### inputs 溯源

1. `train_dataloader = self.get_train_dataloader()` —— 普通情况下
2. `epoch_dataloader = train_dataloader` —— 普通情况
3. `epoch_iterator = iter(epoch_dataloader)`
4. batch_samples 是迭代器 epoch_iterator 的元素组成的列表：
   - `batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)`
   - self.get_batch_samples 方法的源码见：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L5033>
5. inputs 是 batch_samples 的一个元素

#### 支线：Trainer.get_train_dataloader 方法

<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L942>

**注意**：DataLoader 是 torch.utils.data 库的函数

```python
def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
```

其中 `train_dataset = self._remove_unused_columns(train_dataset, description="training")`

**Trainer._remove_unused_columns 方法**（<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L865>）的完整代码如下：

```python
def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
    if not self.args.remove_unused_columns:
        return dataset
    self._set_signature_columns_if_needed()
    signature_columns = self._signature_columns
    
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set"
        logger.info(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            " you can safely ignore this message."
        )
    
    columns = [k for k in signature_columns if k in dataset.column_names]
    if len(columns) == 0:
        raise ValueError(
            "No columns in the dataset match the model's forward method signature. "
            f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
            "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
        )
    
    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        )
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)
```

其中被保留的是：

```python
self._set_signature_columns_if_needed()
signature_columns = self._signature_columns
```

**Trainer._set_signature_columns_if_needed 方法**（<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L850>）全部代码如下：

```python
def _set_signature_columns_if_needed(self):
    if self._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        model_to_inspect = self.model
        if _is_peft_model(self.model):
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = self.model.get_base_model()
            else:
                # PeftMixedModel do not provide a `get_base_model` method
                model_to_inspect = self.model.base_model.model
        signature = inspect.signature(model_to_inspect.forward)
        self._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
```

---

##### 代码修改5

在 `CustomSeq2SeqTrainer` 中重写 `_set_signature_columns_if_needed` 方法，保留 `"loss_weight"`：

```python
@override
def _set_signature_columns_if_needed(self):
    super()._set_signature_columns_if_needed()
    self._signature_columns += ["loss_weight"]
```

---

### 3.3 Trainer.training_step 方法

<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3542>

```python
def training_step(
    self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()

    inputs = self._prepare_inputs(inputs)
    if is_sagemaker_mp_enabled():
        loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        return loss_mb.reduce_mean().detach().to(self.args.device)

    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    del inputs
    if (
        self.args.torch_empty_cache_steps is not None
        and self.state.global_step % self.args.torch_empty_cache_steps == 0
    ):
        if is_torch_xpu_available():
            torch.xpu.empty_cache()
        elif is_torch_mlu_available():
            torch.mlu.empty_cache()
        elif is_torch_musa_available():
            torch.musa.empty_cache()
        elif is_torch_npu_available():
            torch.npu.empty_cache()
        elif is_torch_mps_available(min_version="2.0"):
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

    kwargs = {}

    # For LOMO optimizers you need to explicitly use the learning rate
    if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        kwargs["learning_rate"] = self._get_learning_rate()

    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss *= self.args.gradient_accumulation_steps
        self.accelerator.backward(loss, **kwargs)

    return loss.detach() / self.args.gradient_accumulation_steps
```

#### 支线：Trainer._prepare_inputs 方法 & Trainer._prepare_input 方法

**Trainer._prepare_inputs**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3508>

**Trainer._prepare_input**：<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3490>

`Trainer.training_step` 调用了 `inputs = self._prepare_inputs(inputs)`，`Trainer._prepare_inputs` 中调用了 `inputs = self._prepare_input(inputs)`。

可以看出：
1. `Trainer._prepare_input` 方法首先将输入的 inputs 中的 torch.tensor 转移到 `self.args.device` 设备中，如果采用 deepspeed 还要把浮点数和复数的数格式转化为 `self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()`
2. `Trainer._prepare_inputs` 方法给 inputs 增加新的键值对：`inputs["mems"] = self._past`

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
            # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            # embedding. Other models such as wav2vec2's inputs are already float and thus
            # may need special handling to match the dtypes of the model
            kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
        return data.to(**kwargs)
    return data

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

---

### 3.4 Trainer.compute_loss 方法

<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer.py#L3610>

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

---

### 3.5 LabelSmoother 类

<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/trainer_pt_utils.py#L544>

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

---

### 3.6 CustomSeq2SeqTrainer.compute_loss 方法

<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/trainer.py#L82>

> CustomSeq2SeqTrainer在Trainer的基础上改写了compute_loss方法

```python
@override
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
	r"""
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

---

### 代码修改6

我们修改损失函数的代码，使得能够实现对于不同 `loss_weight` 的样本进行加权。

以下代码均在CustomSeq2SeqTrainer所属文件修改：
<https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/src/llamafactory/train/sft/trainer.py>

#### 导入模块
```python
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from torch import nn
```

#### 重写compute_loss方法

**注意：**
1. 某些 model 的`forward` 方法可能需要输入 labels，则将下面的对应代码替换为 `outputs = model(**inputs, labels=labels)`;
2. 由于版本问题，MODEL_FOR_CAUSAL_LM_MAPPING_NAMES (<https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/models/auto/modeling_auto.py#L462>) 收录的模型可能不全，
根据需要可以将 `elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():` 替换为 `elif (model_name == "模型名") or (model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()):`.

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
	
	r"""
	Fixes the loss value for transformers 4.46.0.
	https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
	"""

	if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
		loss = loss / self.args.gradient_accumulation_steps
		
	return (loss, outputs) if return_outputs else loss
```

#### 2. 增加label_smoother_weighted方法

> 修改自 `LabelSmoother` 类的 `__call__` 方法，方便重写的 `compute_loss` 方法调用。

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




	
