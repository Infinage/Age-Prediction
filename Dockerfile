FROM tensorflow/tensorflow:2.14.0-gpu

# Install libraries we would be using
RUN pip install \
    keras_core keras_cv keras_nlp \
    jupyterlab jupyterlab-vim jupyter-resource-usage \
    tqdm matplotlib numpy pandas opencv-python

# Open CV dependency to be added to Docker file
RUN apt-get update && apt-get install -y libgl1

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Jupyter lab dark mode
RUN mkdir -p "/root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension"
RUN echo "{\"theme\": \"JupyterLab Dark\"}" >> "/root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings"

WORKDIR /app

EXPOSE 8888
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]