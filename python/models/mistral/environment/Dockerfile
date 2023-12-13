FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch211

RUN pip install -U --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip uninstall onnxruntime-training -y
RUN pip install --pre ort-nightly-gpu --extra-index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
RUN pip install transformers
RUN pip install optimum
RUN pip install py3nvml
RUN pip list
