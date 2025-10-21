这是一个非常棒的项目！根据您的需求，以下是您提供的 `README.md` 文档的中文翻译版。

-----

## 工业异常检测中的域偏移：一个真实世界数据集与掩码多尺度重建

\<p align="center"\>
  \<img src=assets/image/mmr.png width="30%"\>
\</p\>

这是论文 [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216) 的官方 PyTorch 实现。

```
@article{zhang2023industrial,
  title={Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction},
  author={Zhang, Zilong and Zhao, Zhibin and Zhang, Xingwu and Sun, Chuang and Chen, Xuefeng},
  journal={arXiv preprint arXiv:2304.02216},
  year={2023}
}
```

### 数据集

我们发布了一个真实世界的**航空发动机叶片异常检测 (Aero-engine Blade Anomaly Detection, AeBAD)** 数据集，包含两个子数据集：**单叶片数据集 (AeBAD-S)** 和**叶片视频异常检测数据集 (AeBAD-V)**。与现有数据集相比，AeBAD 具有以下两个特点：

1.  目标样本**未对齐**且**尺度不同**。
2.  测试集中的正常样本分布与训练集之间存在**域偏移 (domain shift)**，这种域偏移主要是由**光照**和**视角**的变化引起的。

**数据集下载请点击 [此处](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) (Google Drive) 或 [此处](https://cloud.189.cn/web/share?code=nYraE3uMRJn2) (访问代码: g4pr) (天翼云盘)。**

  * **AeBAD-S**

\<p align="center"\>
  \<img src=assets/image/dataset\_s.jpg width="80%"\>
\</p\>

  * **AeBAD-V**

\<p align="center"\>
  \<img src=assets/image/dataset\_v.jpg width="60%"\>
\</p\>

### 视频可视化

①: 原始视频 ②: PatchCore ③: ReverseDistillation ④: DRAEM ⑤: NSA ⑥: MMR (本文方法)

  * **视频 1**

\<table rules="none" align="center"\>
\<tr\>
        \<td\>
\<center\>
\<img src=assets/video/video1/video\_1\_crop.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>①\</font\>
\</center\>
\</td\>
\<td\>
\<center\>
\<img src=assets/video/video1/video\_1\_crop\_PatchCore\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>②\</font\>
\</center\>
\</td\>
\<td\>
\<center\>
\<img src=assets/video/video1/video\_1\_crop\_ReverseDistillation\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>③\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video1/video\_1\_crop\_DRAEM\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>④\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video1/video\_1\_crop\_NSA\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>⑤\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video1/video\_1\_crop\_MMR\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>⑥\</font\>
\</center\>
\</td\>
\</tr\>
\</table\>

  * **视频 2**

\<table rules="none" align="center"\>
\<tr\>
        \<td\>
\<center\>
\<img src=assets/video/video2/video\_2\_crop.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>①\</font\>
\</center\>
\</td\>
\<td\>
\<center\>
\<img src=assets/video/video2/video\_2\_crop\_PatchCore\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>②\</font\>
\</center\>
\</td\>
\<td\>
\<center\>
\<img src=assets/video/video2/video\_2\_crop\_ReverseDistillation\_process.gif width=100 /\>
<br>
\<font color="AAAAAA"\>③\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video2/video\_2\_crop\_DRAEM\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>④\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video2/video\_2\_crop\_NSA\_process.gif width=100 /\>
<br>
\<font color="AAAAAA"\>⑤\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video2/video\_2\_crop\_MMR\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>⑥\</font\>
\</center\>
\</td\>
\</tr\>
\</table\>

  * **视频 3**

\<table rules="none" align="center"\>
\<tr\>
        \<td\>
\<center\>
\<img src=assets/video/video3/video\_3\_crop.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>①\</font\>
\</center\>
\</td\>
\<td\>
\<center\>
\<img src=assets/video/video3/video\_3\_crop\_PatchCore\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>②\</font\>
\</center\>
\</td\>
\<td\>
\<center\>
\<img src=assets/video/video3/video\_3\_crop\_ReverseDistillation\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>③\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video3/video\_3\_crop\_DRAEM\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>④\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video3/video\_3\_crop\_NSA\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>⑤\</font\>
\</center\>
\</td\>
        \<td\>
\<center\>
\<img src=assets/video/video3/video\_3\_crop\_MMR\_process.gif width="100%" /\>
<br>
\<font color="AAAAAA"\>⑥\</font\>
\</center\>
\</td\>
\</tr\>
\</table\>

### 快速开始

#### 预训练模型

在 [此处](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth) 下载 MAE (ViT-base) 的预训练模型。

#### 数据集

**MVTec:**

**创建 MVTec 数据集目录**。从 [此处](https://www.mvtec.com/company/research/datasets/mvtec-ad) 下载 MVTec-AD 数据集。MVTec 数据集目录结构应如下所示。

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- object (bottle, etc.)
                |-- train
                |-- test
                |-- ground_truth
```

**AeBAD:**

从上述链接下载 AeBAD 数据集。AeBAD 数据集目录结构应如下所示。

```
|-- AeBAD
    |-- AeBAD_S
        |-- train
            |-- good
                |-- background
        |-- test
                |-- ablation
                    |-- background
        |-- ground_truth
                |-- ablation
                    |-- view
    |-- AeBAD_V
        |-- test
            |-- video1
                |-- anomaly
        |-- train
            |-- good
                |-- video1_train
```

**请注意，训练集中的背景、视角和光照与测试集不同。测试集中的背景、视角和光照对于训练集是未见过的。**

#### 虚拟环境

使用以下命令：

```
pip install -r requirements.txt
```

#### 针对 MVTec, AeBAD 进行训练和测试

训练模型并针对每个类别或不同域进行评估。这将输出每个类别（或域）的结果（**样本级 AUROC**、**像素级 AUROC** 和 **PRO**）。它还将在目录中生成可视化结果。

运行以下代码：

```
sh mvtec_run.sh
```

```
sh AeBAD_S_run.sh
```

```
sh AeBAD_V_run.sh
```

`MMR.yaml` 中的 `TRAIN.MMR.model_chkpt` 是上述下载模型的路径。`TRAIN.dataset_path` ( `TEST.dataset_path` ) 是数据的路径。
设置 `Test.save_segmentation_images` 为 `True` 或 `False` 来保存处理后的图像。

**请注意，对于 AeBAD-V，我们只评估样本级指标。像素级指标为 0。**

#### 面向工业应用的 GUI

针对工业应用，我们提供了一个基于 PyQt 的图形用户界面（GUI），以便更轻松地使用：

```
python gui_main.py
```

该 GUI 允许您：

  - 加载训练好的模型和配置文件
  - 执行单张图像异常检测
  - 批量处理文件夹中的图像
  - 通过热力图可视化检测结果
  - 调整检测阈值

该界面专为工业环境设计，提供了一种无需命令行交互即可执行异常检测的用户友好方式。

## 致谢

我们感谢 [MAE](https://github.com/facebookresearch/mae) 和 [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) 的出色实现。

## 许可证

本数据集基于 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可证发布。
