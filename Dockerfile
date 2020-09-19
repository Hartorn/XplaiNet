FROM bazire/python:3.7-cpu@sha256:32968fbeb9099625200ebd13759b8e0f1d8560a6ac9f056f1085a0b7a18be3d0

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