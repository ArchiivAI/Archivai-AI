FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install uv

RUN uv pip install --system -r requirements.txt

COPY  ./app /code/app

ENV PYTHONPATH=/code/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
