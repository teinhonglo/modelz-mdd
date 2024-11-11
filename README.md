# WhisperX serving on Mosec

WhisperX splits single audio input into segments to parallel inference,
but this project ONLY focus on using the batch-infernecing enabled **faster-whisper** to serve multiple request.
## Setup
Please follow https://github.com/m-bain/whisperX to install conda env, then

`pip install -r requirements.txt` 

to install `mosec` framework and other packages.
## Usage
```bash
# maximum 12 seconds
python main.py --timeout 12000
```
or
## Build Docker Image

You can choose to use the one of the following method:

* `Dockerfile`
  * `docker buildx build -t <docker_hub_user_name>/<image_name> --push .`
* `build.envd`
  * `envd build -f :whisper_serving --output type=image,name=docker.io/<docker_hub_user_name>/<image_name> --push`

## Test

```
python client.py
```
