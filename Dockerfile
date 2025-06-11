FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install uv
RUN uv pip install -r requirements.txt

COPY  ./app /code/app

ENV PYTHONPATH=/code/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
