## 工业异常检测与领域偏移：一个真实世界的数据集和掩码多尺度重建

<p align="center">
  <img src=assets/image/mmr.png width="30%">
</p>

这是论文《[Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216)》(工业异常检测与领域偏移：一个真实世界的数据集和掩码多尺度重建)的官方 PyTorch 实现。
```
@article{zhang2023industrial,
  title={Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction},
  author={Zhang, Zilong and Zhao, Zhibin and Zhang, Xingwu and Sun, Chuang and Chen, Xuefeng},
  journal={arXiv preprint arXiv:2304.02216},
  year={2023}
}
```

### 数据集

我们发布了一个真实世界的航空发动机叶片异常检测 (AeBAD) 数据集，由两个子数据集组成：单叶片数据集 (AeBAD-S) 和叶片视频异常检测数据集 (AeBAD-V)。与现有数据集相比，AeBAD 具有以下两个特征：1.) 目标样本未对齐且处于不同尺度。2.) 测试集中正常样本的分布与训练集之间存在领域偏移，其中领域偏移主要由光照和视角的变化引起。

**在以下位置下载数据集：[Google Drive](https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view?usp=share_link) 或 [天翼云盘](https://cloud.189.cn/web/share?code=nYraE3uMRJn2) (访问代码: g4pr)。**

* AeBAD-S

<p align="center">
  <img src=assets/image/dataset_s.jpg width="80%">
</p>

* AeBAD-V

<p align="center">
  <img src=assets/image/dataset_v.jpg width="60%">
</p>

### 视频可视化

①: 原始视频 ②: PatchCore ③: ReverseDistillation ④: DRAEM ⑤: NSA ⑥: MMR

* 视频 1

<table rules="none" align="center">
	<tr>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop.gif width="100%" />
				<br/>
				<font color="AAAAAA">①</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video1/video_1_crop_PatchCore_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">②</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video1/video_1_crop_ReverseDistillation_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">③</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop_DRAEM_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">④</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop_NSA_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑤</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video1/video_1_crop_MMR_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑥</font>
			</center>
		</td>
	</tr>
</table>

* 视频 2

<table rules="none" align="center">
	<tr>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop.gif width="100%" />
				<br/>
				<font color="AAAAAA">①</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video2/video_2_crop_PatchCore_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">②</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video2/video_2_crop_ReverseDistillation_process.gif width=100 />
				<br/>
				<font color="AAAAAA">③</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop_DRAEM_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">④</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop_NSA_process.gif width=100 />
				<br/>
				<font color="AAAAAA">⑤</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video2/video_2_crop_MMR_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑥</font>
			</center>
		</td>
	</tr>
</table>

* 视频 3

<table rules="none" align="center">
	<tr>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop.gif width="100%" />
				<br/>
				<font color="AAAAAA">①</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video3/video_3_crop_PatchCore_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">②</font>
			</center>
		</td>
		<td>
			<center>
				<img src=assets/video/video3/video_3_crop_ReverseDistillation_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">③</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop_DRAEM_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">④</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop_NSA_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑤</font>
			</center>
		</td>
        <td>
			<center>
				<img src=assets/video/video3/video_3_crop_MMR_process.gif width="100%" />
				<br/>
				<font color="AAAAAA">⑥</font>
			</center>
		</td>
	</tr>
</table>

### 开始使用

#### 预训练模型

在以下位置下载 MAE (ViT-base) 的预训练模型：[here](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth)。

#### 数据集

**MVTec:**

**创建 MVTec 数据集目录**。从以下位置下载 MVTec-AD 数据集：[here](https://www.mvtec.com/company/research/datasets/mvtec-ad)。MVTec 数据集目录应如下所示：

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

从上述链接下载 AeBAD 数据集。AeBAD 数据集目录应如下所示：

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

**请注意，训练集中的背景、视角和光照与测试集不同。测试集中的背景、视角和光照对于训练集来说是不可见的。**

#### 虚拟环境

使用以下命令：
```
pip install -r requirements.txt
```

#### MVTec 和 AeBAD 的训练与测试

为每个类别或不同领域训练模型并进行评估。这将输出每个类别的结果（样本级 AUROC、像素级 AUROC 和 PRO）。它将在目录中生成可视化结果。

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

MMR.yaml 中的 TRAIN.MMR.model_chkpt 是上述下载模型的路径。TRAIN.dataset_path (TEST.dataset_path) 是数据的路径。
将 Test.save_segmentation_images 设置为 True 或 False 以保存处理后的图像。

**请注意，对于 AeBAD-V，我们仅评估样本级指标。像素级指标为 0。**

#### 工业应用的图形用户界面

为了工业应用，我们提供了一个基于 PyQt 的图形用户界面以便于使用：

```
python gui_main.py
```

该 GUI 允许您：
- 加载训练好的模型和配置文件
- 执行单张图像异常检测
- 处理文件夹中的批量图像
- 使用热图可视化检测结果
- 调整检测阈值

该界面专为工业环境设计，提供了一种用户友好的方式来执行异常检测，无需命令行交互。

## 致谢
我们感谢来自 [MAE](https://github.com/facebookresearch/mae) 和 [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) 的优秀实现。

## 许可证
数据以 CC BY 4.0 许可证发布。
