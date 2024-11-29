# FPredict

(The project is the assignment of 2024 fall DSAA6000M in HKUST-GZ)

In this project, we plan to use amino acid sequences and their associated physical parameters, such as mutations, fluorescence properties and structural data as training dataset for deep learning models, which could be collected from FPbase containing detailed information of 1,073 FPs. Our goal is to predict specific physical parameters of FPs, such as fluorescence colors, intensities and lifetimes. From our model, we would like to screen for brighter or acid-resistant FPs, which have significant potential in single-molecule imaging or correlative light and electron microscopy.

# Dependency

- pandas
- matplotlib
- requests
- json
- tqdm
- biopython

# Record

## download the FPs sequence and relative parameters

- 1006 proteins in total in the data base
- from which, 81 proteins which miss sequence, part of them miss all infomation
- in conclusion, collect **961 json files and 925 sequence** in total

Some proteins need double check manually,

lanRFP-ΔS83l
Pp2FbFPL30M
GFP(S65T)
iRFP713/V256C
eqFP611V124T
AlexaFluorPlus800
ECFPH148D
mKateM41GS158C
R3-2+PCB
mKateS158C
αGFP
Lumazinebindingprotein
SuperfolderYFP
ThermostableGreenProtein
HyperfolderYFP
V127TSAASoti
SuperfoldermTurquoise2
EnhancedCyan-EmittingGFP
SuperfoldermTurquoise2ox
SuperNovaGreen
miniSOGQ103V
E2-Red/Green
cpEYFP(V68L/Q69K)
SuperfolderpHluorin
shBFP-N158S/L173I
dClover2A206K
SuperNovaRed
Trp-lessGFP
mKOκ
spGFP11
shCP-E63L/Y64H
SuperfolderBFP
shBFP-N158K/L173I
pHluorin,ratiometric
Pp2FbFP
SuperfolderCFP
pHluorin,ecliptic
Bovineserumalbumin
cpT-Sapphire174-173
mKateS158A
FoldingReporterGFP
Montiporasp.#20-9115
supereclipticpHluorin
MonomerichyperfolderYFP
SuperfolderGFP

## download spectrum

在数据库中，共包含436个荧光蛋白含有光谱信息，其中有361个蛋白是单独状态的稳定蛋白，其数据已经整理到`./data/spectrum_extracred.json`文件中，其可以使用`pandas`读取

## 数据整合

目前整合得到的数据如下：

| FileName | Content |
| --- | --- |
| `./data/combined_basic.json` | 包含802个有序列信息的荧光蛋白基本性质 |
| `./data/combined_data.json` | 包含343个单状态荧光蛋白的光谱信息 |

使用如下命令载入数据

```python
import pandas as pd
data = pd.read_json("./data/combined_data.json", orient="records", lines=True)
