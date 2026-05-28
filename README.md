# Remote Sensing AI Analysis Demo

遥感科学与技术本科毕业设计成果 — AI + 遥感影像分析 Demo，面试作品集用途。

## 项目结构

| 文件 | 说明 |
|------|------|
| `rs_analysis.py` | 主程序：合成遥感影像生成 + K-Means 土地覆盖分类 + DeepLabV3 语义分割 |
| `scene_classifier.py` | ResNet50 场景分类（ImageNet 预训练） |
| `imagenet_classes.py` | ImageNet 1000 类名称映射表（离线，无需网络） |
| `requirements.txt` | Python 依赖 |

## 快速开始

```bash
pip install -r requirements.txt
python rs_analysis.py
```

无需任何外部数据 — 程序会自动生成合成遥感影像并进行分析。

## 技术方案

### Track 1: K-Means 无监督土地覆盖分类

- 生成模拟真实卫星影像的合成数据（包含水体、森林、农田、城镇、裸地）
- 使用 sklearn K-Means 聚类进行无监督地物分类
- 输出土地利用占比统计 + 可视化图表

### Track 2: DeepLabV3 语义分割

- 使用 COCO 预训练的 DeepLabV3-ResNet50 模型
- 演示深度学习像素级分类 pipeline
- 注：生产环境需使用 DeepGlobe / ISPRS 等遥感数据集微调

### 场景分类

- ResNet50（ImageNet 预训练）迁移学习
- 对输入影像进行 Top-10 场景预测
- 归类到 Urban / Vegetation / Water / Agriculture 等遥感类别

## 输出结果

运行后 `results/` 文件夹将生成：

1. `synthetic_rs_input.png` — 合成的模拟卫星影像
2. `kmeans_landcover.png` — K-Means 土地覆盖分类图
3. `urban_segmentation.png` — DeepLabV3 城市场景分割结果
4. `rs_segmentation.png` — DeepLabV3 遥感影像分割结果

## 依赖

- Python 3.9+
- PyTorch / torchvision
- scikit-learn / matplotlib / Pillow
