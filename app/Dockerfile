FROM tensorflow/tensorflow:2.7.1

LABEL org.opencontainers.image.authors="johns@provoke.co.nz"

COPY . /tf

WORKDIR /tf

RUN chmod +x run_it_all.sh

ENTRYPOINT ["./run_it_all.sh"]