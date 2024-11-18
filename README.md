# An End-to-End MDD Service on Mosec

This project focuses on batch inference with **faster-whisper** to handle multiple requests, unlike WhisperX, which splits a single audio input into segments for parallel inference.

## Setup
To set up the environment, please follow the instructions at [WhisperX GitHub](https://github.com/m-bain/whisperX) (note: WhisperX is not used in this version). First, install the conda environment as per the instructions, then run:

```bash
pip install -r requirements.txt
```

This will install the `mosec` framework and other necessary packages.

## Preparation
1. Download [model.zip](https://140.122.184.167:5567/sharing/qRaWMnSBC) and put it into *models/mdd/*
2. cd models/mdd
3. unzip wav2vec2-mdd.zip


## Usage

Run the following command (with a maximum duration of 12 seconds):

```bash
python app.py --timeout 12000
```

## Build Docker Image

You can choose one of the following methods to build the Docker image:

* Using `Dockerfile`:
  ```bash
  docker buildx build -t <docker_hub_user_name>/<image_name> --push .
  ```

* Using `build.envd`:
  ```bash
  envd build -f :whisper_serving --output type=image,name=docker.io/<docker_hub_user_name>/<image_name> --push
  ```

## Testing

To test the setup, run:

```bash
python client.py
```
