FROM bazire/python:3.9-cpu@sha256:025b66cdedf407d4374df4fd1badbe1d1a9b8a7038e3059797fbf07d6a7c94b6

RUN apt update && apt install curl make git -y
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
SHELL ["/bin/bash", "-lc"]

ENV POETRY_CACHE /work/.cache/poetry
ENV PIP_CACHE_DIR /work/.cache/pip
ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime
ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN $HOME/.poetry/bin/poetry config virtualenvs.path $POETRY_CACHE

ENV PATH ${PATH}:/root/.poetry/bin:/bin:/usr/local/bin:/usr/bin:/usr/sbin:/usr/local/sbin:/sbin

CMD ["bash", "-l"]