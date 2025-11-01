FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3  && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip

# RUN python3 -m pip install --no-cache-dir --upgrade tensorflow[and-cuda] 
# RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RUN python3 -m pip install --no-cache-dir matplotlib

# Install JupyterLab and other necessary packages
# RUN pip3 install --no-cache-dir jupyterlab notebook ipykernel

# RUN python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel nbformat  jedi 
#RUN jupyter serverextension enable --py jupyter_http_over_ws
# RUN python3 -m pip install --no-cache-dir ipywidgets widgetsnbextension pandas-profiling

# COPY requirements.txt /tmp/
# RUN python3 -m pip install --requirement /tmp/requirements.txt

# WORKDIR /

# EXPOSE 8888

