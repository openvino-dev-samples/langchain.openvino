import argparse
from ov_llm import OpenVINO_LLM
from ov_embedding import OpenVINO_Embeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

def main(args):
    
    question = args.question
    llm_model_path = args.llm_model
    embedding_model_path = args.embedding_model
    device = args.device
    template ="""{question}"""
    
    # url = "https://docs.openvino.ai/2023.2/openvino_release_notes.html"
    url = "https://docs.openvino.ai/2023.2/openvino_docs_deployment_optimization_guide_common.html"
 
    print("=== loading web contents ===")
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    print("=== embedding the texts ===")
    embedding = OpenVINO_Embeddings.from_model_id(embedding_model_path)
    text_splitter = CharacterTextSplitter(chunk_size = 200, chunk_overlap = 0)
    texts = text_splitter.split_documents(docs)
    
    print("=== building knowledge database ===")
    # db = Chroma.from_documents(texts, embedding)
    db = FAISS.from_documents(texts, embedding)
    prompt = PromptTemplate(template=template, input_variables=["question"])

    print("=== loading LLM ===")
    llm = OpenVINO_LLM.from_model_id(
        model_id=llm_model_path,
        model_kwargs={"device":device, "temperature": 0, "trust_remote_code": True},
        max_new_tokens=512
    )
    
    print("=== creating chain ===")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4})
    )

    print("=== running pipeline ===")
    respone = chain.run(question)
    print(respone)
    # respone = chain({"query": question}) 
    # print(respone)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVINO LangChain Example')
    parser.add_argument('-e','--embedding_model', type=str, required=False,
                        help='the path to embedding model')
    parser.add_argument('-l','--llm_model', type=str, required=False,
                        help='the path to llm model')
    parser.add_argument('-q', '--question', type=str, default='How to optimize the performance ?',
                        help='qustion you want to ask.')
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='device to run LLM')
    args = parser.parse_args()
    
    main(args)