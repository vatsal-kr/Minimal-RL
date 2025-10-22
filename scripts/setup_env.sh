conda create -n raftpp python=3.12 -y
cd trl
conda activate raftpp
pip install -e .
pip install vllm
pip install --no-build-isolation flash-attn
