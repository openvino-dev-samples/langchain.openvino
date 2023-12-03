import argparse
import bs4
from langchain import hub
from ov_pipeline.llm import OpenVINO_LLM
from ov_pipeline.embedding import OpenVINO_Embeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


def main(args):

    question = args.question
    llm_model_path = args.llm_model
    embedding_model_path = args.embedding_model
    device = args.device

    print("=== processing web documents ===")
    loader = WebBaseLoader(
        web_paths=(
            "https://dlstreamer.github.io/index.html",),
        # bs_kwargs=dict(
        #     parse_only=bs4.SoupStrainer(
        #         class_=("post-content", "post-title", "post-header")
        #     )
        # ),
    )
    loader.requests_kwargs = {'verify': False}
    docs = loader.load()
    print(len(docs[0].page_content))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("=== loading embedding model ===")
    embedding = OpenVINO_Embeddings.from_model_id(embedding_model_path, model_kwargs={
                                                  "device_name": device,  "config": {"PERFORMANCE_HINT": "THROUGHPUT"}})

    print("=== building knowledge database ===")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    retriever = vectorstore.as_retriever()

    print("=== loading LLM ===")
    llm = OpenVINO_LLM.from_model_id(
        model_id=llm_model_path,
        model_kwargs={"device": device,
                      "temperature": 0, "trust_remote_code": True},
        max_new_tokens=512
    )

    print("=== creating RAG chain ===")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain.invoke(question)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVINO LangChain Example')
    parser.add_argument('-e', '--embedding_model', type=str, required=False, default='./embedding_model',
                        help='the path to embedding model')
    parser.add_argument('-l', '--llm_model', type=str, required=False, default='./llm_model',
                        help='the path to llm model')
    parser.add_argument('-q', '--question', type=str, default='What is DL streamer?',
                        help='qustion you want to ask.')
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='device to run LLM')
    args = parser.parse_args()

    main(args)
