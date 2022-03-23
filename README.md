# Summer Project by junzhi Ning


**This is a organized github repositories for the Scientific Research Project in SCIE30001 at UOM**
- Student Name: Junzhi Ning
- Student ID: 1086241
- Report Link: https://www.overleaf.com/4282838782zbddtsztqqxx



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

# Datasets
- External dataset 1: Face-blurred ILSVRC2012–2017 classification -> https://image-net.org/download-images.php
- External dataset 2: Office31 Dataset -> https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code
### PS: Some datasets are not used in the report.

# Directory
- `data`: Contain All preprocessed files and small external datasets supporting the analysis.Usually Store on the local machine to avoid occupying the online storage
- `plots_results`: All plots and outputresults both for model training and reporting writting. This file also contains the storing excel files for data outputing used in the report.
- `code`: Contain all code implementations in this project
  - `testing`: Designed for initial debugging purposes
  -  `version1.0`:Designed for Alexnet model pretraining with selected classes and labels masked
  -  `version2.0`: Designed for UDA fine tuning on pretraind models
  -  `version3.0`: Designed for SSL fine tuning on pretraind models
-  `reports_and_presentation` : Storing the old versions of report and presentation slides for achrived purposes
# References:
See the references in the report link

