import os
from pathlib import Path
import threading

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from loader import load_animation


MODEL_NAME = "qwen2.5-coder:3b"
EMBED_MODEL = "nomic-embed-text"

CODE_EXTENSIONS = {
    ".py", ".java", ".c", ".h", ".cpp", ".hpp",
    ".js", ".ts", ".tsx",
    ".go", ".rs", ".cs", ".php",
    ".scala", ".kt", ".rb", ".sh",
    ".json", ".yaml", ".yml", ".toml",
    ".ini", ".md"
}

IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "target", "build", "dist",
    ".idea", ".vscode"
}

INDEX_DIR = ".agent_index"



def should_index(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in CODE_EXTENSIONS
        and not any(part in IGNORED_DIRS for part in path.parts)
    )


def index_project(root: Path):
    print("Indexation du projet...")

    documents = []
    for path in root.rglob("*"):
        if should_index(path):
            try:
                documents.extend(TextLoader(str(path), encoding="utf-8").load())
            except Exception:
                pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    return Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(root / INDEX_DIR)
    )


def build_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es un assistant expert en analyse de projets logiciels.
Tu peux analyser n'importe quel langage.
Tu réponds UNIQUEMENT à partir du code fourni.
Si l'information n'est pas présente, dis-le clairement.
Quand tu cites du code, indique toujours le fichier.

CONTEXTE:
{context}

QUESTION:
{question}
"""
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return chain


def main():
    root = Path(os.getcwd())
    print(f"Projet courant: {root}")

    vectordb = index_project(root)
    chain = build_chain(vectordb)

    print("Ready. Tape 'exit' pour quitter.")

    while True:
        question = input("\n❓ Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        done =[False]
        t = threading.Thread(target=load_animation(), args=(done,))
        response = chain.invoke(question)
        done[0]=True
        t.join()

        print("\nRéponse:\n")
        print(response.content)


if __name__ == "__main__":
    main()

