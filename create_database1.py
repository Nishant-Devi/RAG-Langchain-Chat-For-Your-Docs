from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from dotenv import load_dotenv
import os
import shutil
from openai import OpenAI
import openai

load_dotenv()
openai.api_key=os.environ["OPENAI_API_KEY"]

Data_Path = "data/books"
Chroma_Path = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# Turn the files into sth called a document
"""A document is going to contain all of the content
and it's also going to contain a bunch of metadata"""
def load_documents():
    loader = DirectoryLoader(Data_Path, glob="*.md")
    documents = loader.load()
    return documents

"""A single document can be really really long. So it's not enough that
we load each markdown file into one document. We split this big document
into smaller chunks. When we search through all of this data, each chunk
is going to be more focused and more relevant to what we are looking for."""
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap=500,
        length_function =len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document= chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

"""To be able to query each chunk, we are going to need to turn
this into a database. We will be using ChromaDB for this, which is a special
kind of database that uses vector embeddings as the key.
For this we need an OPENAI account beacuse we are going to use the OPENAI embeddings
function to generate the vector emebeddings for each chunk"""

# Embeddings are vector representation of text that capture their meaning. In Python, this is literally a list of numbers.
# To actually generate a vector from a word, we will need an LLM, like OpenAI. And this is usually just an API or a function we can call.

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(Chroma_Path):
        shutil.rmtree(Chroma_Path)
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=Chroma_Path
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {Chroma_Path}")

if __name__ == "__main__":
    main()
