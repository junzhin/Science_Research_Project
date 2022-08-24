# Summer Project by junzhi Ning


**This is a organized github repository for the Scientific Research Project in SCIE30001 at UOM**
** 分析半监督与无监督迁移学习计算机视觉算法异同点，提出预训练神经网络对模型影响的可能性

# Dependencies:
- Language: i.e Python 3.7.11
- Packages / Libraries:
  - pandas
  - torch>=1.7.0
  - torchvision
  - qpsolvers
  - numpy
  - prettytable
  - tqdm
  - scikit-learn
  - webcolors  
  - matplotlib
  - torch>=1.7.0
  - Transfer library github from https://github.com/thuml/Transfer-Learning-Library
  - libraries from https://github.com/YBZh/Bridging_UDA_SSL and https://github.com/YBZh/Bridging_UDA_SSL

# Datasets
- External dataset 1: Face-blurred ILSVRC2012–2017 classification -> https://image-net.org/download-images.php
- External dataset 2: Office31 Dataset -> https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code
### PS: Some datasets are not used in the report.

# Directory
- `data`: Contain All preprocessed files and small external datasets supporting the analysis.Usually Store on the local machine to avoid occupying the online storage
- `plots_results`: All plots and outputresults both for model training and reporting writting. This file also contains the storing excel files for data outputing used in the report.
- `code`: Contain all code implementations in this project
  - `testing3`: Designed for initial debugging purposes
  -  `version1.0`: Designed for Alexnet model pretraining with selected classes and labels masked
  -  `version2.0`: Designed for UDA fine tuning on pretrained models
  -  `version3.0`: Designed for SSL fine tuning on pretrained models
-  `reports_and_presentation` : Storing the old versions of report and presentation slides for archived purposes.
# References:
See the references in the report link

