from http import HTTPStatus

import requests
from datasets import load_dataset

import msgpack  # type: ignore


def main():
    url = "http://localhost:8000/inference"
    sample = "./example/1272-128104-0002.wav"
    print(sample)
    with open(sample, "rb") as f:
        binary = f.read()
        
    req = {
        "binary": binary,
        "id": "1",
    }
    resp = requests.post(url, data=msgpack.packb(req))
    if resp.status_code == HTTPStatus.OK:
        print(resp.content.decode("utf-8"))
    else:
        print(resp.status_code, resp.text)


if __name__ == "__main__":
    main()
