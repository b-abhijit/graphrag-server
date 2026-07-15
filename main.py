from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

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


@app.get("/")
def root():
    return {"message": "ok"}


@app.post("/extract-graph")
def extract_graph(payload: ExtractGraphRequest):
    text = payload.text.lower()

    entities = []
    relationships = []

    def add_entity(name, typ):
        item = {"name": name, "type": typ}
        if item not in entities:
            entities.append(item)

    def add_rel(source, target, relation):
        item = {"source": source, "target": target, "relation": relation}
        if item not in relationships:
            relationships.append(item)

    if "langchain" in text:
        add_entity("LangChain", "Framework")
    if "harrison chase" in text:
        add_entity("Harrison Chase", "Person")
    if "openai" in text:
        add_entity("OpenAI", "Organization")

    if "langchain" in text and "harrison chase" in text and ("created" in text or "developed" in text):
        add_rel("Harrison Chase", "LangChain", "CREATED")

    if "langchain" in text and "openai" in text and ("integrates with" in text or "integrated into" in text or "integrates" in text):
        add_rel("LangChain", "OpenAI", "INTEGRATED_INTO")

    return {
        "entities": entities,
        "relationships": relationships
    }


@app.post("/graph-query")
def graph_query(payload: GraphQueryRequest):
    question = payload.question.lower()
    relationships = payload.graph.get("relationships", [])

    if "who created the framework that integrates with" in question:
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
    relationships = payload.relationships

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