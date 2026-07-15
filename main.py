from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import re

app = FastAPI(title="GraphRAG Server")


# ----------------------------
# Request models
# ----------------------------

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


# ----------------------------
# Helper data
# ----------------------------

KNOWN_ENTITIES = {
    "Harrison Chase": "Person",
    "OpenAI": "Organization",
    "LangChain": "Framework",
    "GraphMind Systems": "Organization",
    "LlamaIndex": "Framework",
    "Anthropic": "Organization",
    "ChatGPT": "Product",
    "Claude": "Product",
}


def add_entity(entities, name, entity_type):
    if not any(e["name"] == name for e in entities):
        entities.append({"name": name, "type": entity_type})


def add_relationship(relationships, source, target, relation):
    relationships.append({
        "source": source,
        "target": target,
        "relation": relation
    })


# ----------------------------
# Health/root route
# ----------------------------

@app.get("/")
def root():
    return {"message": "GraphRAG server is running"}


# ----------------------------
# 1. Extract graph
# ----------------------------

@app.post("/extract-graph")
def extract_graph(payload: ExtractGraphRequest):
    text = payload.text
    entities = []
    relationships = []

    # Detect known entities
    for name, entity_type in KNOWN_ENTITIES.items():
        if name.lower() in text.lower():
            add_entity(entities, name, entity_type)

    # Pattern: "X was created by Y"
    m = re.search(r"([A-Za-z0-9\s]+?) was created by ([A-Za-z\s]+)", text, re.IGNORECASE)
    if m:
        target = m.group(1).strip()
        source = m.group(2).strip().rstrip(".")
        add_entity(entities, target, KNOWN_ENTITIES.get(target, "Framework"))
        add_entity(entities, source, KNOWN_ENTITIES.get(source, "Person"))
        add_relationship(relationships, source, target, "CREATED")

    # Pattern: "X created Y"
    m = re.search(r"([A-Za-z\s]+) created ([A-Za-z0-9\s]+)", text, re.IGNORECASE)
    if m:
        source = m.group(1).strip()
        target = m.group(2).strip().rstrip(".")
        add_entity(entities, source, KNOWN_ENTITIES.get(source, "Person"))
        add_entity(entities, target, KNOWN_ENTITIES.get(target, "Framework"))
        add_relationship(relationships, source, target, "CREATED")

    # Pattern: "X founded Y"
    m = re.search(r"([A-Za-z\s]+) founded ([A-Za-z0-9\s]+)", text, re.IGNORECASE)
    if m:
        source = m.group(1).strip()
        target = m.group(2).strip().rstrip(".")
        add_entity(entities, source, KNOWN_ENTITIES.get(source, "Person"))
        add_entity(entities, target, KNOWN_ENTITIES.get(target, "Organization"))
        add_relationship(relationships, source, target, "FOUNDED")

    # Pattern: "X developed Y"
    m = re.search(r"([A-Za-z\s]+) developed ([A-Za-z0-9\s]+)", text, re.IGNORECASE)
    if m:
        source = m.group(1).strip()
        target = m.group(2).strip().rstrip(".")
        add_entity(entities, source, KNOWN_ENTITIES.get(source, "Person"))
        add_entity(entities, target, KNOWN_ENTITIES.get(target, "Product"))
        add_relationship(relationships, source, target, "DEVELOPED")

    # Pattern: "X integrates with Y"
    m = re.search(r"([A-Za-z0-9\s]+) integrates with ([A-Za-z0-9\s]+)", text, re.IGNORECASE)
    if m:
        source = m.group(1).strip()
        target = m.group(2).strip().rstrip(".")
        add_entity(entities, source, KNOWN_ENTITIES.get(source, "Framework"))
        add_entity(entities, target, KNOWN_ENTITIES.get(target, "Organization"))
        add_relationship(relationships, source, target, "INTEGRATED_INTO")

    return {
        "entities": entities,
        "relationships": relationships
    }


# ----------------------------
# 2. Graph query
# ----------------------------

@app.post("/graph-query")
def graph_query(payload: GraphQueryRequest):
    question = payload.question.lower()
    entities = payload.graph.get("entities", [])
    relationships = payload.graph.get("relationships", [])

    # Example template:
    # "Who created the framework that integrates with OpenAI?"
    if "who created the framework that integrates with" in question:
        target_name = payload.question.split("with")[-1].strip().rstrip("?")

        framework = None
        creator = None

        for rel in relationships:
            if rel["relation"] == "INTEGRATED_INTO" and rel["target"].lower() == target_name.lower():
                framework = rel["source"]
                break

        if framework:
            for rel in relationships:
                if rel["target"].lower() == framework.lower() and rel["relation"] in ["CREATED", "DEVELOPED"]:
                    creator = rel["source"]
                    break

        if framework and creator:
            return {
                "answer": creator,
                "reasoning_path": [target_name, framework, creator],
                "hops": 2
            }

    return {
        "answer": "Answer not found",
        "reasoning_path": [],
        "hops": 0
    }


# ----------------------------
# 3. Community summary
# ----------------------------

@app.post("/community-summary")
def community_summary(payload: CommunitySummaryRequest):
    entity_counts = {}

    for rel in payload.relationships:
        entity_counts[rel["source"]] = entity_counts.get(rel["source"], 0) + 1
        entity_counts[rel["target"]] = entity_counts.get(rel["target"], 0) + 1

    center = max(entity_counts, key=entity_counts.get) if entity_counts else (payload.entities[0] if payload.entities else "Unknown")

    phrases = []

    for rel in payload.relationships:
        if rel["target"] == center and rel["relation"] == "CREATED":
            phrases.append(f"created by {rel['source']}")
        elif rel["source"] == center and rel["relation"] == "INTEGRATED_INTO":
            phrases.append(f"integrates with {rel['target']}")
        elif rel["target"] == center and rel["relation"] == "FOUNDED":
            phrases.append(f"founded by {rel['source']}")

    if phrases:
        summary = f"This community centers around {center}, which is " + " and ".join(phrases) + "."
    else:
        summary = f"This community centers around {center} and its connected relationships."

    return {
        "community_id": payload.community_id,
        "summary": summary
    }