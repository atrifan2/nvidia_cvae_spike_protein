# for admininstration of a running container:
# $ docker exec -it --user root [container] bash
#
APIKEY=~/NGC.NVIDIA.COM.API
REGISTRY_USER='$$oauthtoken'
REGISTRY=nvcr.io
REG_PATH=${REGISTRY}/nvidian/sae/astern
DOCKERFILE=Dockerfile
CONTAINER_NAME=horovod_chem_2

PROJECT_DIR=/home/abe/projects/devrel-Bhowmik
CONTAINER_HOME=/projects

HOST_JUPYTER_PORT=9000
HOST_TB_PORT=9001

USER=$(shell id -u)
GID=$(shell id -g)

.PHONY : dockerlogin build all

all: build

dockerlogin:
	docker login ${REGISTRY} --username=${REGISTRY_USER} --password=$(shell cat ~/NGC.NVIDIA.COM.API)

build: dockerlogin
	docker build -f ${DOCKERFILE} \
                     -t ${CONTAINER_NAME} .
	echo done

tag: build
	docker tag ${CONTAINER_NAME} ${REG_PATH}:${CONTAINER_NAME}

push: tag
	docker push ${REG_PATH}:${CONTAINER_NAME}

pull: dockerlogin
	docker pull ${REG_PATH}:${CONTAINER_NAME}

run:

	nvidia-docker run -it \
        --publish ${HOST_JUPYTER_PORT}:8888 \
        --publish ${HOST_TB_PORT}:6006 \
        --env USER=${USER} \
        --env HOME=${CONTAINER_HOME} \
        --user ${USER}:${GID} \
        --volume /etc/group:/etc/group \
        --volume /etc/passwd:/etc/passwd \
        --volume ${PROJECT_DIR}:/projects \
        --workdir /projects \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        ${REG_PATH}:${CONTAINER_NAME}

jupyter:

	nvidia-docker run -it \
        --publish ${HOST_JUPYTER_PORT}:8888 \
        --publish ${HOST_TB_PORT}:6006 \
        --env USER=${USER} \
        --env HOME=${CONTAINER_HOME} \
        --user ${USER}:${GID} \
        --volume /etc/group:/etc/group \
        --volume /etc/passwd:/etc/passwd \
        --volume ${PROJECT_DIR}:/projects \
        --workdir /projects \
        --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 \
        ${REG_PATH}:${CONTAINER_NAME} /projects/run_jupyter.sh notebook

