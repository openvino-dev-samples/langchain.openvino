# langchain.openvino
This experimental sample shows how to implement embedding and text generation model with OpenVINO runtime and LangChain.

## Requirements

- Linux, Windows
- Python >= 3.9.0
- CPU or GPU compatible with OpenVINO.
- RAM: >= 16GB
- vRAM: >= 8GB

## Install the requirements

    $ python3 -m venv openvino_env

    $ source openvino_env/bin/activate

    $ python3 -m pip install --upgrade pip
    
    $ pip install wheel setuptools
    
    $ pip install -r requirements.txt


## Run LLM sample

Export the LLM IR model from HuggingFace (optional):

    $ python3 model_export/export_llm.py -m "meta-llama/Llama-2-7b-chat-hf"

Run text generation sample with local models:

    $ python3 sample.py -m "./ir_model"

Run text generation sample with remote models:

    $ python3 sample.py -m "meta-llama/Llama-2-7b-chat-hf"

## Run RAG sample

Export the embedding IR model from HuggingFace:

    $ python3 model_export/export_embedding.py

Export the LLM IR model from HuggingFace (optional):

    $ python3 model_export/export_llm.py -m "meta-llama/Llama-2-7b-chat-hf"

Run RAG sample with local models:

    $ python3 rag.py
