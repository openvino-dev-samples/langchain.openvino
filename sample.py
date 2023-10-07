import argparse

from transformersllm import TransformersLLM
from langchain import PromptTemplate, LLMChain


def main(args):
    
    question = args.question
    model_path = args.model_path
    device = args.device
    template ="""{question}"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    # llm = TransformersPipelineLLM.from_model_id(
    #     model_id=model_path,
    #     task="text-generation",
    #     model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True},
    # )

    llm = TransformersLLM.from_model_id(
        model_id=model_path,
        model_kwargs={"device":device, "temperature": 0, "max_length": 64, "trust_remote_code": True},
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = llm_chain.run(question)
    print("====output=====")
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransformersLLM Langchain Chat Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='What is AI?',
                        help='qustion you want to ask.')
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='device to run LLM')
    args = parser.parse_args()
    
    main(args)