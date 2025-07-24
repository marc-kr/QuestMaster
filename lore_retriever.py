from typing import List

from langchain_core.vectorstores import VectorStoreRetriever, InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

class LoreRetriever:
    def __init__(self, lore_file_path, search_type='similarity', k=5, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):

        self.lore_file_path = lore_file_path

        with open(lore_file_path, 'r', encoding='utf-8') as lore_file:
            self.lore_text = lore_file.read()
        self.retriever = setup_retriever(self.lore_text, k=k, search_type=search_type, embedding_model_name=embedding_model_name)

    def retrieve(self, query: str):
        """Retrieve relevant lore for the quest description"""


        return self.retriever.invoke(query)

            #lore_context = "\n\n".join([doc.page_content for doc in docs])


def setup_retriever(document: str, k=5, search_type='similarity',
                    embedding_model_name='sentence-transformers/all-MiniLM-L6-v2') -> VectorStoreRetriever:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n\n', '\n', ' ', '']
    )
    chunks: List[str] = text_splitter.split_text(document)
    embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = InMemoryVectorStore.from_texts(chunks, embedding=embedder)
    return vector_store.as_retriever(search_kwargs={"k": k}, search_type=search_type)


if __name__ == '__main__':
    # Example usage
    lore_retriever = LoreRetriever('./testing/quest_scifi.txt')
    query = "What is the history of the ancient ruins?"
    lore_context = lore_retriever.retrieve(query)
    print("Retrieved Lore Context:\n", lore_context)



