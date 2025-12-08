# 1. 基础环境 (锁定 Python 3.11)
conda create -p /home/yyj/opt/anaconda3/envs/m python=3.11 -y
conda activate m

# 2. 科学计算与 GIS (基于代码引用)
conda install numpy pandas matplotlib scipy rasterio -y
pip install opencv-python pyyaml tqdm

# 3. 深度学习框架
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 torchmetrics==1.8.2

# 4. 核心库版本锁定 (基于报错信息)
pip install lightning==2.5.6

# 5. 模型相关依赖 (基于 import 分析)
pip install timm tensorboard "pydantic>=2.0"

# 6. Mamba 模块 (必须安装，否则 simvp_module.py 报错)
pip install mamba-ssm