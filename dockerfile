FROM python:3.9
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /opt/
WORKDIR /opt
ADD venv /users/maximiliantronich/PycharmProjects/DataScience/venv
CMD ["python", "Model.py"]
