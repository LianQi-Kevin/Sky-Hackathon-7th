sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
pip3 install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install --user pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install rosa numpy==1.19.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torchmetrics==0.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install nemo_toolkit[all]==1.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install ASR-metrics -i https://pypi.tuna.tsinghua.edu.cn/simple