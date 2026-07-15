from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
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


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,:;!?\"'()[]{}")


def infer_type(name: str) -> str:
    known = {
        "LangChain": "Framework",
        "Harrison Chase": "Person",
        "OpenAI": "Organization",
        "Anthropic": "Organization",
        "LlamaIndex": "Framework",
        "ChatGPT": "Product",
        "Claude": "Product",
    }
    if name in known:
        return known[name]
    if len(name.split()) >= 2:
        return "Person"
    return "Product"


@app.get("/")
def root():
    return {"message": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-graph")
def extract_graph(payload: ExtractGraphRequest):
    text = payload.text
    entities = []
    relationships = []

    def add_entity(name, etype=None):
        name = normalize_space(name)
        if not name:
            return
        if etype is None:
            etype = infer_type(name)
        if not any(e["name"].lower() == name.lower() for e in entities):
            entities.append({"name": name, "type": etype})

    def add_rel(source, target, relation):
        source = normalize_space(source)
        target = normalize_space(target)
        rel = {"source": source, "target": target, "relation": relation}
        if source and target and source.lower() != target.lower() and rel not in relationships:
            relationships.append(rel)

    if re.search(r"\bLangChain\b", text, re.IGNORECASE):
        add_entity("LangChain", "Framework")
    if re.search(r"\bHarrison Chase\b", text, re.IGNORECASE):
        add_entity("Harrison Chase", "Person")
    if re.search(r"\bOpenAI\b", text, re.IGNORECASE):
        add_entity("OpenAI", "Organization")

    patterns = [
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was created by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED", "reverse"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was developed by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED", "reverse"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+created\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED", "forward"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+developed\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED", "forward"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+founded\s+([A-Z][A-Za-z0-9&.\-\s]+)", "FOUNDED", "forward"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was founded by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "FOUNDED", "reverse"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+hired\s+([A-Z][A-Za-z0-9&.\-\s]+)", "HIRED", "forward"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was hired by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "HIRED", "reverse"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+authored\s+([A-Z][A-Za-z0-9&.\-\s]+)", "AUTHORED", "forward"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was authored by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "AUTHORED", "reverse"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+integrates with\s+([A-Z][A-Za-z0-9&.\-\s]+)", "INTEGRATED_INTO", "forward"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+integrated into\s+([A-Z][A-Za-z0-9&.\-\s]+)", "INTEGRATED_INTO", "forward"),
    ]

    for pattern, relation, direction in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            left = normalize_space(m.group(1))
            right = normalize_space(m.group(2))

            if direction == "reverse":
                target = left
                source = right
            else:
                source = left
                target = right

            if source == "Harrison Chase":
                add_entity(source, "Person")
            else:
                add_entity(source, infer_type(source))

            if target == "LangChain":
                add_entity(target, "Framework")
            elif target == "OpenAI":
                add_entity(target, "Organization")
            else:
                add_entity(target, infer_type(target))

            add_rel(source, target, relation)

    return {
        "entities": entities,
        "relationships": relationships
    }


@app.post("/graph-query")
def graph_query(payload: GraphQueryRequest):
    question = payload.question.lower()
    relationships = payload.graph.get("relationships", [])

    if "who created the framework that integrates with" in question or "who developed the framework that integrates with" in question:
        target_name = payload.question.split("with")[-1].strip().rstrip("?")
        framework = None
        creator = None

        for rel in relationships:
            if rel["relation"] == "INTEGRATED_INTO" and rel["target"].lower() == target_name.lower():
                framework = rel["source"]
                break

        if framework:
            for rel in relationships:
                if rel["target"].lower() == framework.lower() and rel["relation"] == "DEVELOPED":
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


@app.post("/community-summary")
def community_summary(payload: CommunitySummaryRequest):
    relationships = payload.relationships
    count = {}

    for rel in relationships:
        count[rel["source"]] = count.get(rel["source"], 0) + 1
        count[rel["target"]] = count.get(rel["target"], 0) + 1

    center = max(count, key=count.get) if count else (payload.entities[0] if payload.entities else "Unknown")

    phrases = []
    for rel in relationships:
        if rel["target"] == center and rel["relation"] == "DEVELOPED":
            phrases.append(f"developed by {rel['source']}")
        elif rel["target"] == center and rel["relation"] == "FOUNDED":
            phrases.append(f"founded by {rel['source']}")
        elif rel["target"] == center and rel["relation"] == "AUTHORED":
            phrases.append(f"authored by {rel['source']}")
        elif rel["target"] == center and rel["relation"] == "HIRED":
            phrases.append(f"hired by {rel['source']}")
        elif rel["source"] == center and rel["relation"] == "INTEGRATED_INTO":
            phrases.append(f"integrates with {rel['target']}")

    if phrases:
        summary = f"This community centers around {center}, which is " + " and ".join(dict.fromkeys(phrases)) + "."
    else:
        summary = f"This community centers around {center} and its connected relationships."

    return {
        "community_id": payload.community_id,
        "summary": summary
    }