# 严格配对消融实验（B / Q / Z / Q+Z / Q+Z+P）执行方案

为了在论文中彻底解决因果归因与模块相互作用问题，拟在 **3 个代表性骨干网络**上，按照统一控制变量口径（同 Parent 权重、同学习率 Schedule、同候选输入）全面执行 `B` $\rightarrow$ `Q` $\rightarrow$ `Z` $\rightarrow$ `Q+Z` $\rightarrow$ `Q+Z+P` 严格配对实验。

---

## 一、骨干架构选择（Backbones）

1. **Moment-DETR**（经典的 DETR 架构时间定位基线）
2. **EaTR**（Event-aware Temporal 迁移骨干）
3. **Flash-VTG**（现代强基线 Transformer 骨干）

---

## 二、消融矩阵设计 (5×3 = 15 个配置)

每个骨干网络均评估以下 5 种递进配置：

| 配置符号 | 模块组成 | 核心目的 |
|:---:|---|---|
| **B** | Baseline GMR | 建立统一基准线 |
| **Q** | B + Quality Head | 验证定位与边界排序净收益 |
| **Z** | B + Independent Zero Head | 验证独立第二层判空复核能力 |
| **Q+Z** | B + Quality + Zero | 验证定位模块与判空模块是否有负交互 |
| **Q+Z+P** | B + Quality + Zero + Learned Dedup | 评估完整框架 U 的集合级最终性能 |

---

## 三、GPU 资源分配与执行流水线

由于当前系统配备 **2 张 NVIDIA GeForce RTX 3090 (24GB VRAM)**：

- **GPU 0**：并行执行 **Moment-DETR** 矩阵 (B, Q, Z, Q+Z) 及后续 Selector 训练；
- **GPU 1**：并行执行 **EaTR** 矩阵与 **Flash-VTG** 矩阵。

---

## 四、严格控制变量规范 (Controlled-Variable Protocol)

1. **Parent 冻结/统一**：每个骨干的所有子变体必须继承相同的 Parent Checkpoint；
2. **学习率 schedule 匹配**：新 Head 以 3e-5 训练，现有权重保持统一微调步数；
3. **测试集封存**：所有选择和消融仅在 Standard Validation 上进行，确定最佳模型后统一盲测 Test 集；
4. **报告指标**：同时报告 AUROC、Rej-F1@0.4、mAP、mR@5、mR+@5、G-mIoU@3。

