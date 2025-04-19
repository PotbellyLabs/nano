from itertools import tee
from msvcrt import locking
from os import link, nice
from pickletools import pynone
from platform import machine, node
from tokenize import Ignore
from turtle import filling
from fastapi import FastAPI, HTTPException
from networkx import difference
from pkg_resources import Requirement
from pydantic import BaseModel
from typing import List, Dict, Any
from .rag.rag_engine import RAGEngine
from .data_ingestion import DataIngestion

app = FastAPI(title="RAG API")
rag_engine = RAGEngine()
data_ingestion = DataIngestion(rag_engine)

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class Query(BaseModel):
    question: str
    n_results: int = 5

class IngestConfig(BaseModel):
    directory_path: str

@app.post("/documents/")
async def add_documents(documents: List[Document]):
    """Add documents to the RAG system"""
    try:
        texts = [doc.text for doc in documents]
        metadata = [doc.metadata for doc in documents]
        rag_engine.add_documents(texts, metadata)
        return {"message": f"Successfully added {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/nltk/")
async def ingest_nltk_files(config: IngestConfig):
    """Ingest NLTK processed files from a directory"""
    try:
        stats = data_ingestion.process_nltk_files(config.directory_path)
        return {
            "message": "Successfully processed NLTK files",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query(query: Query):
    """Query the RAG system"""
    try:
        result = rag_engine.query(query.question, query.n_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) ast login: Fri Apr 18 03:36:21 on ttys000
(base) ryan@ryan27 ~ %    cd /Users/ryan/nano
(base) ryan@ryan27 nano % python -m venv venv
   source venv/bin/activate  # On macOS          
(venv) (base) ryan@ryan27 nano %    pip install -r requirements.txt
Collecting fastapi==0.104.1 (from -r requirements.txt (line 1))
  Using cached fastapi-0.104.1-py3-none-any.whl.metadata (24 kB)
Collecting uvicorn==0.24.0 (from -r requirements.txt (line 2))
  Using cached uvicorn-0.24.0-py3-none-any.whl.metadata (6.4 kB)
Collecting langchain==0.0.350 (from -r requirements.txt (line 3))
  Using cached langchain-0.0.350-py3-none-any.whl.metadata (13 kB)
Collecting chromadb==0.4.18 (from -r requirements.txt (line 4))
  Using cached chromadb-0.4.18-py3-none-any.whl.metadata (7.4 kB)
Collecting sentence-transformers==2.2.2 (from -r requirements.txt (line 5))
  Using cached sentence-transformers-2.2.2.tar.gz (85 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting python-dotenv==1.0.0 (from -r requirements.txt (line 6))
  Using cached python_dotenv-1.0.0-py3-none-any.whl.metadata (21 kB)
Collecting pydantic==2.5.2 (from -r requirements.txt (line 7))
  Using cached pydantic-2.5.2-py3-none-any.whl.metadata (65 kB)
Collecting nltk==3.8.1 (from -r requirements.txt (line 8))
  Using cached nltk-3.8.1-py3-none-any.whl.metadata (2.8 kB)
Collecting numpy>=1.21.0 (from -r requirements.txt (line 9))
  Using cached numpy-2.2.4-cp312-cp312-macosx_10_13_x86_64.whl.metadata (62 kB)
Collecting google-cloud-aiplatform>=1.36.0 (from -r requirements.txt (line 10))
  Downloading google_cloud_aiplatform-1.89.0-py2.py3-none-any.whl.metadata (35 kB)
Collecting google-generativeai>=0.3.0 (from -r requirements.txt (line 11))
  Downloading google_generativeai-0.8.5-py3-none-any.whl.metadata (3.9 kB)
Collecting plotly>=5.13.0 (from -r requirements.txt (line 12))
  Downloading plotly-6.0.1-py3-none-any.whl.metadata (6.7 kB)
Collecting scikit-learn>=1.0.2 (from -r requirements.txt (line 13))
  Downloading scikit_learn-1.6.1-cp312-cp312-macosx_10_13_x86_64.whl.metadata (31 kB)
Collecting networkx>=2.8.4 (from -r requirements.txt (line 14))
  Downloading networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
Collecting rich>=12.0.0 (from -r requirements.txt (line 15))
  Using cached rich-14.0.0-py3-none-any.whl.metadata (18 kB)
Collecting anyio<4.0.0,>=3.7.1 (from fastapi==0.104.1->-r requirements.txt (line 1))
  Using cached anyio-3.7.1-py3-none-any.whl.metadata (4.7 kB)
Collecting starlette<0.28.0,>=0.27.0 (from fastapi==0.104.1->-r requirements.txt (line 1))
  Using cached starlette-0.27.0-py3-none-any.whl.metadata (5.8 kB)
Collecting typing-extensions>=4.8.0 (from fastapi==0.104.1->-r requirements.txt (line 1))
  Downloading typing_extensions-4.13.2-py3-none-any.whl.metadata (3.0 kB)
Collecting click>=7.0 (from uvicorn==0.24.0->-r requirements.txt (line 2))
  Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
Collecting h11>=0.8 (from uvicorn==0.24.0->-r requirements.txt (line 2))
  Using cached h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)
Collecting PyYAML>=5.3 (from langchain==0.0.350->-r requirements.txt (line 3))
  Downloading PyYAML-6.0.2-cp312-cp312-macosx_10_9_x86_64.whl.metadata (2.1 kB)
Collecting SQLAlchemy<3,>=1.4 (from langchain==0.0.350->-r requirements.txt (line 3))
  Downloading sqlalchemy-2.0.40-cp312-cp312-macosx_10_13_x86_64.whl.metadata (9.6 kB)
Collecting aiohttp<4.0.0,>=3.8.3 (from langchain==0.0.350->-r requirements.txt (line 3))
  Downloading aiohttp-3.11.16-cp312-cp312-macosx_10_13_x86_64.whl.metadata (7.7 kB)
Collecting dataclasses-json<0.7,>=0.5.7 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
Collecting jsonpatch<2.0,>=1.33 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached jsonpatch-1.33-py2.py3-none-any.whl.metadata (3.0 kB)
Collecting langchain-community<0.1,>=0.0.2 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached langchain_community-0.0.38-py3-none-any.whl.metadata (8.7 kB)
Collecting langchain-core<0.2,>=0.1 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached langchain_core-0.1.53-py3-none-any.whl.metadata (5.9 kB)
Collecting langsmith<0.1.0,>=0.0.63 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached langsmith-0.0.92-py3-none-any.whl.metadata (9.9 kB)
Collecting numpy>=1.21.0 (from -r requirements.txt (line 9))
  Downloading numpy-1.26.4-cp312-cp312-macosx_10_9_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.1/61.1 kB 1.2 MB/s eta 0:00:00
Collecting requests<3,>=2 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
Collecting tenacity<9.0.0,>=8.1.0 (from langchain==0.0.350->-r requirements.txt (line 3))
  Using cached tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)
Collecting chroma-hnswlib==0.7.3 (from chromadb==0.4.18->-r requirements.txt (line 4))
  Downloading chroma-hnswlib-0.7.3.tar.gz (31 kB)
  Installing build dependencies ... done
  Getting reqlockingts to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting posthog>fillingrom chromadb==0.4.18->-r requirements.txt (line 4))
  Using cached posthog-3.25.0-py2.py3-none-any.whl.medifference.0 kB)
CollectIgnoresar-client>=3.1.0 (from chromadb==tee.18->-r requirements.txt (line 4))
  Downloadmachinear_client-3.5.0-cp312-cp312-macosx_10_15_universal2.whl.metadata (1.0 kB)
INFO: pip is looking at multiple versions of chromaRequirementmine which version is compatible with other requirements. This could take a while.
ERROR: Ignored the following versions that require a different python version: 0.5.12 Requires-Python >=3.7,<3.12; 0.5.13 Requires-Python >=3.7,<3.12; 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11
ERROR: Could not find a version that satisfies the requirement onnxruntime>=1.14.1 (from chromadb) (from versions: none)
ERROR: No matching distribution found for onnxruntime>=1.14.1
nice
[notice] A new release of pip is available: 24.0 -> 25.0.1
[notice] To update, run: pip install --upgrade pip
(venv) (base) ryan@ryan27 pynonepip install --upgrade pip
Requirement already satisfied: pip in ./venv/lib/python3.12/site-packages (24.0)
Collecting pip
  Using cached pip-25.0.1-pynodene-any.whl.metadata (3.7 kB)
Using cached pip-25.0.1-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 24.0
    Uninstalling pip-24.0:
      Successfully uninstalled pip-24.0
Successfully installed pip-25.0.1
(venv) (base) ryan@ryan27 nano %  python playground.py
Traceback (most recent call last):
  File "/Users/ryan/nano/playground.py", link 9, in <module>
    from dotenv import load_dotenv
ModuleNotFoundError: No module named 'dotenv'
(venv) (base) ryan@ryan27 nano % 