FROM python:3-slim-stretch

RUN pip install --upgrade pip
RUN apt-get update; apt-get -y install wget git jq
RUN wget https://github.com/jgm/pandoc/releases/download/2.9.1.1/pandoc-2.9.1.1-1-amd64.deb
RUN dpkg -i pandoc-2.9.1.1-1-amd64.deb
RUN pip install jupyter watchdog[watchmedo] jupyter_client ipykernel jupyter
RUN pip install torch torchvision
RUN python3 -m ipykernel install --user
RUN pip install nbdev

WORKDIR /fastpages
COPY . .
RUN chmod u+x action_entrypoint.sh
RUN chmod u+x word2post.sh

CMD [ "/fastpages/action_entrypoint.sh" ]
