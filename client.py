from http import HTTPStatus
import requests
import msgpack  # type: ignore
import json

def main():
    url = "http://localhost:8000/inference"
    sample = "./example/Something_good_just_happened.wav"
    print(sample)
    
    with open(sample, "rb") as f:
        binary = f.read()

    req = {
        "binary": binary,
        "id": "1",
        "prompt_words": "Something good just happened",
        "prompt_phones": "Something good just happened",
        "do_g2p": True
    }

    resp = requests.post(url, data=msgpack.packb(req))

    if resp.status_code == HTTPStatus.OK:
        # Decode the response content and parse it as JSON
        response_json = json.loads(resp.content.decode("utf-8"))
        
        # Extract the "transcript" field
        print(response_json)
        transcript = response_json.get("transcript", None)
        if transcript:
            print("Transcript:", transcript)
        else:
            print("Transcript field not found in the response.")
    else:
        print(resp.status_code, resp.text)

if __name__ == "__main__":
    main()
