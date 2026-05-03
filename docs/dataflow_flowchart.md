<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --text: #111827;
    --text-sub: #6B7280;
    --text-muted: #9CA3AF;
    --bg: #FFFFFF;
    --surface: #F9FAFB;
    --border: #E5E7EB;
    --blue: #3B82F6;
    --cyan: #06B6D4;
    --connector: #94A3B8;
    --accent: #2563EB;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'SimHei', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }

  #root {
    width: fit-content;
    min-width: 860px;
    margin: 0 auto;
    padding: 48px 40px 60px;
  }

  /* ── Title ── */
  .chart-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    text-align: center;
    margin-bottom: 6px;
  }
  .chart-subtitle {
    font-size: 13px;
    color: var(--text-sub);
    text-align: center;
    margin-bottom: 36px;
  }

  /* ── Phase group ── */
  .phase-group {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 20px 24px 18px;
    margin-bottom: 0;
    position: relative;
  }
  .phase-title {
    font-size: 15px;
    font-weight: 700;
    padding: 9px 16px;
    border-radius: 8px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .phase-title .phase-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.7);
    color: inherit;
  }

  /* ── Phase color scheme: blue-gray progression (7 phases) ── */
  .phase-1 .phase-title { background: #F0F4F8; color: #334155; border-left: 4px solid #64748B; }
  .phase-1 .step-num { background: #E2E8F0; color: #475569; }
  .phase-1 .fn-node { border-color: #CBD5E1; }

  .phase-2 .phase-title { background: #EDF1F5; color: #2D3F54; border-left: 4px solid #5B7A99; }
  .phase-2 .step-num { background: #DBEAFE; color: #1E3A5F; }
  .phase-2 .fn-node { border-color: #B8C9DB; }

  .phase-3 .phase-title { background: #E9EEF3; color: #253650; border-left: 4px solid #4A6B8A; }
  .phase-3 .step-num { background: #D0D9E4; color: #1E3050; }
  .phase-3 .fn-node { border-color: #A8BDCF; }

  .phase-4 .phase-title { background: #E5EBF1; color: #1D2E45; border-left: 4px solid #3A5C7A; }
  .phase-4 .step-num { background: #C7D2E0; color: #172540; }
  .phase-4 .fn-node { border-color: #98AFc3; }

  .phase-5 .phase-title { background: #E1E8EF; color: #182740; border-left: 4px solid #2E5070; }
  .phase-5 .step-num { background: #BCCAD8; color: #152035; }
  .phase-5 .fn-node { border-color: #8AA2B8; }

  .phase-6 .phase-title { background: #DDE5ED; color: #142035; border-left: 4px solid #244565; }
  .phase-6 .step-num { background: #B0C0D0; color: #121C30; }
  .phase-6 .fn-node { border-color: #7C96AE; }

  .phase-7 .phase-title { background: #D9E2EB; color: #101A2E; border-left: 4px solid #1A3A5A; }
  .phase-7 .step-num { background: #A4B6C8; color: #0F1828; }
  .phase-7 .fn-node { border-color: #6E8AA4; }

  /* ── Steps inside phase ── */
  .phase-steps {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-left: 12px;
  }
  .phase-step {
    font-size: 13px;
    font-weight: 400;
    color: var(--text);
    padding: 7px 14px;
    background: white;
    border-radius: 6px;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.5;
  }
  .step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 20px;
    height: 20px;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
  }
  .fn-name {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 12.5px;
    font-weight: 600;
    color: #1E3A5F;
    background: #EFF6FF;
    padding: 1px 7px;
    border-radius: 4px;
    white-space: nowrap;
  }
  .fn-node {
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .step-desc {
    font-size: 12px;
    color: var(--text-sub);
  }
  .step-arrow {
    color: var(--connector);
    font-size: 12px;
    margin: 0 2px;
    flex-shrink: 0;
  }

  /* ── Phase connector arrow ── */
  .phase-connector {
    text-align: center;
    color: #94A3B8;
    font-size: 20px;
    line-height: 1;
    padding: 6px 0;
  }
  .phase-connector-label {
    font-size: 11px;
    color: var(--text-muted);
    display: block;
    margin-top: 1px;
  }

  /* ── Legend ── */
  .flow-legend {
    display: flex;
    gap: 24px;
    justify-content: center;
    margin-top: 32px;
    padding: 14px 24px;
    background: #F9FAFB;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    flex-wrap: wrap;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #4B5563;
  }
  .legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    border: 2px solid;
  }
  .legend-fn {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #1E3A5F;
    background: #EFF6FF;
    padding: 1px 6px;
    border-radius: 3px;
  }

  /* ── Modification badge ── */
  .mod-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 8px;
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
    white-space: nowrap;
    flex-shrink: 0;
  }

  /* ── Data flow tag ── */
  .data-tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 500;
    padding: 1px 6px;
    border-radius: 8px;
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #BFDBFE;
    white-space: nowrap;
    flex-shrink: 0;
  }

  /* ── Branch layout for parallel paths ── */
  .branch-row {
    display: flex;
    gap: 8px;
    align-items: stretch;
  }
  .branch-row .phase-step {
    flex: 1;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  .branch-label {
    font-size: 10px;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 2px;
  }
</style>
</head>
<body>
<div id="root">

  <div class="chart-title">LLaMA-Factory 样本级损失加权 — 完整数据流</div>
  <div class="chart-subtitle">追踪 loss_weight 字段从原始 JSON 到损失计算的完整传递路径（共 6 个修改点）</div>

  <!-- ═══════ Phase 1: 训练入口 ═══════ -->
  <div class="phase-group phase-1">
    <div class="phase-title">
      <span class="phase-badge">Phase 1</span>
      训练入口
      <span class="data-tag">workflow.py</span>
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">1</span>
        <span class="fn-name fn-node">run_sft()</span>
        <span class="step-desc">SFT 训练工作流入口，串联数据加载、Trainer 初始化、训练执行</span>
      </div>
    </div>
  </div>

  <div class="phase-connector">
    ↓
    <span class="phase-connector-label">调用 get_dataset() 加载数据</span>
  </div>

  <!-- ═══════ Phase 2: 数据准备 ═══════ -->
  <div class="phase-group phase-2">
    <div class="phase-title">
      <span class="phase-badge">Phase 2</span>
      数据准备：解析配置与注册字段
      <span class="data-tag">parser.py</span>
      <span class="mod-badge">修改点 1</span>
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">2</span>
        <span class="fn-name fn-node">get_dataset()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">_get_merged_dataset()</span>
        <span class="step-desc">顶层入口，触发数据集加载与融合</span>
      </div>
      <div class="phase-step">
        <span class="step-num">3</span>
        <span class="fn-name fn-node">_get_merged_dataset()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">get_dataset_list()</span>
        <span class="step-desc">读取 dataset_info.json，为每个数据集创建 DatasetAttr 对象</span>
      </div>
      <div class="phase-step">
        <span class="step-num">4</span>
        <span class="fn-name fn-node">DatasetAttr</span>
        <span class="step-desc">数据集属性数据类，通过 set_attr() 解析 columns 配置</span>
        <span class="mod-badge">修改点 1: 增加 loss_weight 属性</span>
      </div>
    </div>
  </div>

  <div class="phase-connector">
    ↓
    <span class="phase-connector-label">逐个加载数据集文件</span>
  </div>

  <!-- ═══════ Phase 3: 数据加载与格式对齐 ═══════ -->
  <div class="phase-group phase-3">
    <div class="phase-title">
      <span class="phase-badge">Phase 3</span>
      数据加载与格式对齐
      <span class="data-tag">loader.py / aligner.py</span>
      <span class="mod-badge">修改点 2</span>
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">5</span>
        <span class="fn-name fn-node">_load_single_dataset()</span>
        <span class="step-desc">使用 HF load_dataset() 加载原始 JSON 文件</span>
      </div>
      <div class="phase-step">
        <span class="step-num">6</span>
        <span class="fn-name fn-node">_load_single_dataset()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">align_dataset()</span>
        <span class="step-desc">根据 formatting 选择转换函数</span>
      </div>
      <div class="phase-step">
        <span class="step-num">7</span>
        <span class="fn-name fn-node">align_dataset()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">convert_sharegpt()</span>
        <span class="step-desc">将 sharegpt 格式转为标准格式（_prompt, _response, _system ...）</span>
        <span class="mod-badge">修改点 2: 输出 _loss_weight 字段</span>
      </div>
      <div class="phase-step">
        <span class="step-num">8</span>
        <span class="fn-name fn-node">merge_dataset()</span>
        <span class="step-desc">将多个对齐后的数据集融合为一个</span>
      </div>
    </div>
  </div>

  <div class="phase-connector">
    ↓
    <span class="phase-connector-label">对融合数据集进行 Tokenization</span>
  </div>

  <!-- ═══════ Phase 4: 数据预处理 ═══════ -->
  <div class="phase-group phase-4">
    <div class="phase-title">
      <span class="phase-badge">Phase 4</span>
      数据预处理：Tokenization 与权重传递
      <span class="data-tag">loader.py / preprocess.py / supervised.py</span>
      <span class="mod-badge">修改点 3 &amp; 4</span>
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">9</span>
        <span class="fn-name fn-node">_get_preprocessed_dataset()</span>
        <span class="step-desc">预处理入口，通过 dataset.map() 批量处理</span>
      </div>
      <div class="phase-step">
        <span class="step-num">10</span>
        <span class="fn-name fn-node">get_preprocess_and_print_func()</span>
        <span class="step-desc">根据 stage 和 packing 配置路由到对应预处理函数</span>
      </div>

      <!-- Branch: non-packed vs packed -->
      <div class="branch-row">
        <div class="phase-step">
          <div class="branch-label">非打包模式</div>
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span class="fn-name fn-node">preprocess_supervised_dataset()</span>
            <span class="step-arrow">→</span>
            <span class="fn-name fn-node">_encode_supervised_example()</span>
          </div>
          <span class="mod-badge">修改点 3: 追加 loss_weight 到 model_inputs</span>
        </div>
        <div class="phase-step">
          <div class="branch-label">打包模式</div>
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span class="fn-name fn-node">preprocess_packed_supervised_dataset()</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:2px;">
            <span class="step-arrow">→</span>
            <span class="fn-name fn-node">_encode_supervised_example()</span>
            <span class="step-arrow">→</span>
            <span class="fn-name fn-node">greedy_knapsack()</span>
          </div>
          <span class="mod-badge">修改点 4: 拼接 packed_loss_weight</span>
        </div>
      </div>
    </div>
  </div>

  <div class="phase-connector">
    ↓
    <span class="phase-connector-label">构建 DataLoader 送入训练</span>
  </div>

  <!-- ═══════ Phase 5: DataLoader 构建 ═══════ -->
  <div class="phase-group phase-5">
    <div class="phase-title">
      <span class="phase-badge">Phase 5</span>
      DataLoader 构建与列过滤
      <span class="data-tag">trainer.py (Trainer)</span>
      <span class="mod-badge">修改点 5</span>
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">11</span>
        <span class="fn-name fn-node">CustomSeq2SeqTrainer.__init__()</span>
        <span class="step-desc">初始化 Trainer，接收预处理后的 dataset_module</span>
      </div>
      <div class="phase-step">
        <span class="step-num">12</span>
        <span class="fn-name fn-node">get_train_dataloader()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">_remove_unused_columns()</span>
        <span class="step-desc">过滤不在模型 forward 签名中的列</span>
      </div>
      <div class="phase-step">
        <span class="step-num">13</span>
        <span class="fn-name fn-node">_set_signature_columns_if_needed()</span>
        <span class="step-desc">通过 inspect.signature 获取模型 forward 参数列表</span>
        <span class="mod-badge">修改点 5: 追加 "loss_weight" 到签名列</span>
      </div>
    </div>
  </div>

  <div class="phase-connector">
    ↓
    <span class="phase-connector-label">进入训练循环</span>
  </div>

  <!-- ═══════ Phase 6: 训练执行 ═══════ -->
  <div class="phase-group phase-6">
    <div class="phase-title">
      <span class="phase-badge">Phase 6</span>
      训练执行：前向传播与损失计算
      <span class="data-tag">trainer.py (Trainer)</span>
      <span class="mod-badge">修改点 6</span>
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">14</span>
        <span class="fn-name fn-node">trainer.train()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">_inner_training_loop()</span>
        <span class="step-desc">启动训练，进入内层循环</span>
      </div>
      <div class="phase-step">
        <span class="step-num">15</span>
        <span class="fn-name fn-node">training_step()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">_prepare_inputs()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">_prepare_input()</span>
        <span class="step-desc">单步训练：将输入张量移至 GPU</span>
      </div>
      <div class="phase-step">
        <span class="step-num">16</span>
        <span class="fn-name fn-node">CustomSeq2SeqTrainer.compute_loss()</span>
        <span class="step-desc">重写的损失计算方法，注入 loss_weight</span>
        <span class="mod-badge">修改点 6: 实现加权损失</span>
      </div>
      <div class="phase-step">
        <span class="step-num">17</span>
        <span class="fn-name fn-node">label_smoother_weighted()</span>
        <span class="step-arrow">→</span>
        <span class="fn-name fn-node">model.forward()</span>
        <span class="step-desc">加权标签平滑 + 模型前向传播，输出加权损失值</span>
      </div>
    </div>
  </div>

  <div class="phase-connector">
    ↓
    <span class="phase-connector-label">梯度累积 / 多 GPU 同步</span>
  </div>

  <!-- ═══════ Phase 7: 梯度处理 ═══════ -->
  <div class="phase-group phase-7">
    <div class="phase-title">
      <span class="phase-badge">Phase 7</span>
      梯度累积与多 GPU 行为
    </div>
    <div class="phase-steps">
      <div class="phase-step">
        <span class="step-num">18</span>
        <span class="fn-name fn-node">loss.backward()</span>
        <span class="step-desc">反向传播，梯度累积至 gradient_accumulation_steps 步后执行 optimizer.step()</span>
      </div>
      <div class="phase-step">
        <span class="step-num">19</span>
        <span class="fn-name fn-node">accelerator.backward()</span>
        <span class="step-desc">多 GPU 场景下由 Accelerate 库处理梯度同步（DDP / DeepSpeed）</span>
      </div>
    </div>
  </div>

  <!-- ── Legend ── -->
  <div class="flow-legend">
    <div class="legend-item">
      <div class="legend-dot" style="border-color:#64748B;background:#F0F4F8;"></div>
      阶段分组
    </div>
    <div class="legend-item">
      <span class="legend-fn">function()</span>
      函数名称
    </div>
    <div class="legend-item">
      <span class="mod-badge">修改点 N</span>
      需要修改的代码位置
    </div>
    <div class="legend-item">
      <span class="data-tag">file.py</span>
      所在源文件
    </div>
    <div class="legend-item">
      <span style="color:#94A3B8;font-size:14px;">→</span>
      调用关系
    </div>
  </div>

</div>
</body>
</html>
