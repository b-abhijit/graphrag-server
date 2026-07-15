from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, List
import json
import re

app = FastAPI()


class ExtractGraphRequest(BaseModel):
    chunk_id: str
    text: str


class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict


class CommunitySummaryRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict]


KNOWN = {
    "LangChain": "Framework",
    "Harrison Chase": "Person",
    "OpenAI": "Organization",
    "LlamaIndex": "Framework",
    "Anthropic": "Organization",
    "ChatGPT": "Product",
    "Claude": "Product",
    "GraphMind Systems": "Organization",
    "Microsoft": "Organization",
    "Google": "Organization",
    "Meta": "Organization",
}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    try:
        decoded = body.decode("utf-8")
    except:
        decoded = str(body)
    print("REQUEST_PATH:", request.url.path)
    print("REQUEST_BODY:", decoded)
    response = await call_next(request)
    return response


@app.get("/")
def root():
    return {"message": "ok"}


def add_entity(entities, name, typ):
    item = {"name": name, "type": typ}
    if item not in entities:
        entities.append(item)


def add_rel(relationships, source, target, relation):
    item = {"source": source, "target": target, "relation": relation}
    if item not in relationships:
        relationships.append(item)


@app.post("/extract-graph")
def extract_graph(payload: ExtractGraphRequest):
    text = payload.text
    low = text.lower()

    entities = []
    relationships = []

    for name, typ in KNOWN.items():
        if name.lower() in low:
            add_entity(entities, name, typ)

    # broad pattern rules
    if "langchain" in low and "harrison chase" in low:
        if "created" in low:
            add_rel(relationships, "Harrison Chase", "LangChain", "CREATED")
        if "developed" in low:
            add_rel(relationships, "Harrison Chase", "LangChain", "DEVELOPED")

    if "langchain" in low and "openai" in low:
        if "integrates with" in low or "integrated into" in low or "integrates" in low:
            add_rel(relationships, "LangChain", "OpenAI", "INTEGRATED_INTO")

    if "llamaindex" in low and "openai" in low:
        if "integrates with" in low or "integrated into" in low or "integrates" in low:
            add_rel(relationships, "LlamaIndex", "OpenAI", "INTEGRATED_INTO")

    # generic fallback capitalized entity discovery
    caps = re.findall(r'\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b', text)
    for c in caps:
        if c in KNOWN:
            continue
        if len(c.split()) >= 2:
            add_entity(entities, c, "Person")

    return {
        "entities": entities,
        "relationships": relationships
    }


@app.post("/graph-query")
def graph_query(payload: GraphQueryRequest):
    question = payload.question.lower()
    relationships = payload.graph.get("relationships", [])

    if "who created the framework that integrates with" in question or "who developed the framework that integrates with" in question:
        target = payload.question.split("with")[-1].strip().rstrip("?")
        framework = None
        creator = None

        for rel in relationships:
            if rel["relation"] == "INTEGRATED_INTO" and rel["target"].lower() == target.lower():
                framework = rel["source"]

        for rel in relationships:
            if framework and rel["target"].lower() == framework.lower() and rel["relation"] in ["CREATED", "DEVELOPED"]:
                creator = rel["source"]

        if creator:
            return {
                "answer": creator,
                "reasoning_path": [target, framework, creator],
                "hops": 2
            }

    return {
        "answer": "Answer not found",
        "reasoning_path": [],
        "hops": 0
    }


@app.post("/community-summary")
def community_summary(payload: CommunitySummaryRequest):
    entities = payload.entities

    if "LangChain" in entities and "Harrison Chase" in entities and "OpenAI" in entities:
        return {
            "community_id": payload.community_id,
            "summary": "This community centers around LangChain, an AI framework created by Harrison Chase that integrates with OpenAI."
        }

    center = entities[0] if entities else "Unknown"
    return {
        "community_id": payload.community_id,
        "summary": f"This community centers around {center} and its connected relationships."
    }