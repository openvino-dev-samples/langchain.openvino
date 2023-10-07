# langchain.openvino
This sample shows how to implement a text generation model with OpenVINO runtime and LangChain.

The part of codes refered to [BigDL](https://github.com/intel-analytics/BigDL/tree/main/python)

## Requirements

- Linux, Windows, MacOS
- Python >= 3.8.0
- CPU or GPU compatible with OpenVINO.
- RAM: 32GB
- vRAM: >=16GB

## Install the requirements

    $ python3 -m venv openvino_env

    $ source openvino_env/bin/activate

    $ python3 -m pip install --upgrade pip
    
    $ pip install wheel setuptools
    
    $ pip install -r requirements.txt


## Run Sample

Export the IR model from HuggingFace (optional):

    $ python3 export.py -m "meta-llama/Llama-2-7b-chat-hf"

Run text generation sample with local models:

    $ python3 sample.py -m "./ir_model"

Run text generation sample with remote models:

    $ python3 sample.py -m "meta-llama/Llama-2-7b-chat-hf"