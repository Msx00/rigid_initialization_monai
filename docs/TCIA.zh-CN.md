> Run installation and CLI commands from the combined repository root.

# TCIA 前列腺 MR→US 刚性配准与 MONAI 数据导出

将原生 MR 刚性配准到 US，以 US 腺体质心为中心，输出 **175 × 175 × 175、0.8 mm 各向同性**的 MR、US、腺体与可选病灶标签。按患者保留已有的 train/val/test 划分，生成路径相对于数据集根目录的 `dataset.json`。[English](TCIA.md)

## 安装并运行合成示例

在仓库根目录、Python 3.10 或更新版本的虚拟环境中执行：

```bash
python -m pip install ".[tcia]"
tcia-rigid demo --output outputs/demo
tcia-rigid batch --monai --pairs outputs/demo/input/pairs.csv --output outputs/example
```

打开 `outputs/example/dataset.json`，在 `test` 中可找到一对完整的合成数据。配准使用 CPU。需要参考实验的软件版本时，在 Python 3.11 环境使用 `python -m pip install -c requirements-tcia-tested.txt ".[tcia,dicom]"`。

## 导出自己的 train/val/test

```bash
tcia-rigid batch --monai \
  --pairs registration_dataset/pairs.csv --data-root registration_dataset \
  --output outputs/MONAI_TCIA_Rigid_Alignment --workers 8
```

输入为**尚未配准的原生图像**及二值前列腺标签，支持 NRRD 和 NIfTI。每个模态内部的图像和标签必须共用其原始物理网格。`pairs.csv` 每行一对，填写 `pair_id`、`patient`、`split`（train、val 或 test）、`mr_image`、`us_image`、`mr_prostate_label` 和 `us_prostate_label`。独立病灶可通过编号传入；匹配编号用于 TRE。路径相对于 `--data-root`。多次检查或多个配准对属于同一患者时，填写同一个患者 ID。[完整字段说明](tcia/DATA_FORMAT.md)

从官方 DICOM、STL 和配对表开始时，安装 `python -m pip install '.[tcia,dicom]'`，使用 `tcia-rigid prepare` 转换。若已有官方或研究使用的划分，直接使用相应配对表。需要新划分时执行：

```bash
tcia-rigid split --pairs data/prepared/pairs.csv --output data/splits --ratios .6 .1 .3 --seed 42
tcia-rigid batch --monai --pairs data/splits/pairs.csv --data-root data/prepared --output outputs/registered --workers 8
```

按患者重新划分后，成员和配准对数量可能与参考实验不同。根据可用内存调节 `--workers`。运行时查看 `progress.json`；中断后用相同参数重跑，会核对输入、版本和文件校验值，并复用合格的病例。错误明细见 `summary.json`。[原始数据准备](tcia/RAW_TCIA.md)

## TCIA 默认配置与输出

默认 **PCA 旋转比例 0、平滑 0 次、最多 4000 个表面采样点、ICP 最多 150 次**，网格简化 20%。初始化对齐腺体网格质心；PCA 为 0 时，ICP 仍会估计旋转和平移。拟合使用原生毫米坐标下的前列腺表面。图像线性插值，标签最近邻插值。超过 30° 的变换标记为待复查。可用 `--config` 或命令行覆盖默认值。[参数文件](../configs/tcia.json)

| 输出 | 内容 |
|---|---|
| `dataset.json` | `training`、`validation`、`test` 三个列表；记录中的文件路径相对于清单所在目录 |
| `<split>/<patient>/<pair_id>/mr/` 和 `us/` | 共网格图像、腺体标签，以及输入中提供的独立和合并病灶标签 |
| 每对的 `.tfm`、`registration.json` | 原生 MR→US 的 LPS 毫米变换、供重采样的逆变换、参数、指标、复查标记和校验记录 |
| `export_check.json`、`metrics/*.csv` | 各划分汇总、逐对及逐靶点误差；患者等权 TRE 和配准对等权 Dice |

输出 MR 已应用变换，可直接与同一对的 US 一起读取。安装 MONAI 后，按 `mr_image`、`us_image` 和相应标签键构造输入字典即可。[MONAI 加载示例](TCIA.md#output-and-loading)

完整验证集 124 对的患者平均靶点 TRE 为 **7.32 mm**，腺体 Dice 为 **0.8293**；独立测试集 373 对分别为 **7.47 mm** 和 **0.8233**。TRE 仅统计对应病灶，不包含腺体质心。[统计口径和全部结果](tcia/EXPERIMENTS.md)

## 测试和许可

安装后运行 `python -m pip install '.[tcia,dicom]'
python -m unittest discover -s tests -v`。本仓库提供代码与合成示例，真实数据从原始 TCIA 集合获取。代码采用 Apache-2.0，网格、PCA 与 ICP 的上游来源和版本见 [NOTICE](../NOTICE) 和 [UPSTREAM.json](../UPSTREAM-TCIA.json)。
