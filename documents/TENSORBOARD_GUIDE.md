# TensorBoard 使用指南

## 📊 功能概述

训练脚本已集成 PyTorch TensorBoard 功能，可以实时可视化训练过程中的各项指标。

---

## 🎯 记录的指标

### 1. **训练损失（Train_Step）** - 每个训练步记录
- `Train_Step/l2` - L2损失
- `Train_Step/freq` - 频域损失
- `Train_Step/stat` - 统计特征损失
- `Train_Step/total` - 总损失
- `Train_Step/learning_rate` - 学习率

### 2. **训练损失（Train_Epoch）** - 每个epoch平均值
- `Train_Epoch/l2` - 平均L2损失
- `Train_Epoch/freq` - 平均频域损失
- `Train_Epoch/stat` - 平均统计特征损失
- `Train_Epoch/total` - 平均总损失
- `Train_Epoch/learning_rate` - 当前学习率

### 3. **验证损失（Validation）** - 验证时记录
- `Validation/l2` - 验证L2损失
- `Validation/freq` - 验证频域损失
- `Validation/stat` - 验证统计特征损失
- `Validation/total` - 验证总损失

---

## 🚀 使用方法

### 1. 启动训练（自动记录）

```bash
cd /Users/frozen2001/PycharmProjects/PythonProject
python -m SR.train_noise_predictor --config SR/configs/train_noise_predictor.yaml
```

训练时会自动记录所有指标到 `SR/experiments/noise_predictor/tensorboard/` 目录。

### 2. 启动 TensorBoard 可视化

在**新的终端窗口**中运行：

```bash
cd /Users/frozen2001/PycharmProjects/PythonProject
tensorboard --logdir=SR/experiments/noise_predictor/tensorboard
```

然后在浏览器中打开：`http://localhost:6006`

### 3. 指定端口（如果6006被占用）

```bash
tensorboard --logdir=SR/experiments/noise_predictor/tensorboard --port=6007
```

### 4. 远程服务器使用

如果在远程服务器训练，需要端口转发：

```bash
# 在本地机器执行
ssh -L 6006:localhost:6006 user@remote_server

# 然后在远程服务器启动tensorboard
tensorboard --logdir=SR/experiments/noise_predictor/tensorboard
```

在本地浏览器访问：`http://localhost:6006`

---

## 📈 TensorBoard 界面说明

### SCALARS（标量）标签页

- **Train_Step**: 查看每个训练步的详细损失曲线（更细粒度）
- **Train_Epoch**: 查看每个epoch的平均损失曲线（更平滑）
- **Validation**: 查看验证集上的损失变化

### 常用功能

1. **平滑曲线**：左侧滑块调整平滑度（Smoothing）
2. **对比实验**：可以同时加载多个实验目录对比
3. **下载数据**：点击左下角下载按钮导出CSV数据
4. **缩放**：鼠标滚轮缩放，拖拽平移

---

## 🔍 监控重点

### 训练是否正常

✅ **正常训练的特征**：
- 总损失（total）持续下降
- 学习率按照调度器正常变化
- 验证损失与训练损失趋势一致

⚠️ **需要注意的情况**：
- 损失突然爆炸（NaN或Inf）→ 学习率过大
- 损失不下降 → 学习率过小或模型问题
- 验证损失上升但训练损失下降 → 过拟合

### 各项损失的意义

- **L2损失**：像素级重建质量
- **频域损失**：频率域特征匹配
- **统计特征损失**：统计特性一致性
- **总损失**：加权总和，主要优化目标

---

## 📁 文件结构

```
SR/experiments/noise_predictor/
├── tensorboard/           # TensorBoard日志目录
│   └── events.out.tfevents.*  # 事件文件
├── checkpoints/          # 模型权重
└── samples/              # 验证样本
```

---

## 🛠️ 高级用法

### 1. 对比多个实验

```bash
# 同时加载多个实验
tensorboard --logdir_spec=exp1:SR/experiments/exp1/tensorboard,exp2:SR/experiments/exp2/tensorboard
```

### 2. 只查看特定指标

在TensorBoard界面左侧搜索框输入关键词，如 `total` 只显示总损失。

### 3. 导出数据进行分析

点击左下角的下载按钮，可以导出CSV格式的数据用于进一步分析。

---

## ❓ 常见问题

### Q1: TensorBoard 显示 "No dashboards are active"

**原因**：训练还没有开始记录数据，或者路径不正确。

**解决**：
1. 确认训练已经开始
2. 检查 `--logdir` 路径是否正确
3. 刷新浏览器页面

### Q2: 曲线不更新

**原因**：TensorBoard 缓存问题。

**解决**：
1. 点击右上角刷新按钮
2. 或者重启 TensorBoard

### Q3: 端口被占用

**错误信息**：`Address already in use`

**解决**：
```bash
# 使用其他端口
tensorboard --logdir=SR/experiments/noise_predictor/tensorboard --port=6007
```

### Q4: 如何清除旧的日志

```bash
# 删除旧的TensorBoard日志（谨慎操作！）
rm -rf SR/experiments/noise_predictor/tensorboard/*
```

---

## 💡 最佳实践

1. **训练前清理旧日志**：避免混淆不同实验的数据
2. **使用有意义的实验名称**：在配置文件中设置清晰的 `save_dir`
3. **定期查看TensorBoard**：及时发现训练问题
4. **保存重要实验的日志**：备份 `tensorboard` 目录
5. **结合checkpoint使用**：损失下降时对应的checkpoint最有价值

---

## 📚 参考资源

- [TensorBoard 官方文档](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard 教程](https://pytorch.org/docs/stable/tensorboard.html)

---

## ✅ 验证安装

训练完成后，终端会显示：

```
TensorBoard日志已保存到: SR/experiments/noise_predictor/tensorboard
使用以下命令查看: tensorboard --logdir=SR/experiments/noise_predictor/tensorboard
```

按照提示启动 TensorBoard 即可查看训练过程！🎉