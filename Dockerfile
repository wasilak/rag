FROM python:3

RUN pip install uv

ADD . /app

WORKDIR /app

RUN uv sync

ENV USER_AGENT="CLIzilla/3.7 (🤖 still learning; may or may not eat your RAM; report bugs to mom)"

ENTRYPOINT ["uv", "run", "main.py"]

CMD []
