from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import re

app = FastAPI(title="GraphRAG API", version="1.0")


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


KNOWN_ENTITIES = {
    "Harrison Chase": "Person",
    "OpenAI": "Organization",
    "LangChain": "Framework",
    "LlamaIndex": "Framework",
    "Anthropic": "Organization",
    "ChatGPT": "Product",
    "Claude": "Product",
    "GraphMind Systems": "Organization",
    "Microsoft": "Organization",
    "Google": "Organization",
    "Meta": "Organization",
}


REL_ALIASES = {
    "CREATED": "DEVELOPED",
    "BUILT": "DEVELOPED",
    "WROTE": "AUTHORED",
    "WRITTEN_BY": "AUTHORED",
    "INTEGRATES_WITH": "INTEGRATED_INTO",
    "WORKS_WITH": "INTEGRATED_INTO",
}

ALLOWED_TYPES = {"Person", "Organization", "Product", "Framework"}
ALLOWED_RELATIONS = {"FOUNDED", "DEVELOPED", "INTEGRATED_INTO", "HIRED", "AUTHORED"}


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,:;!?\"'()[]{}")


def canonical_relation(rel: str) -> str:
    rel = rel.upper().strip()
    return REL_ALIASES.get(rel, rel)


def infer_type(name: str) -> str:
    if name in KNOWN_ENTITIES:
        return KNOWN_ENTITIES[name]
    if any(tok in name for tok in ["Inc", "Corp", "Systems", "Labs", "AI", "Company", "Organization"]):
        return "Organization"
    if any(tok in name for tok in ["Chain", "Index", "Framework", "SDK"]):
        return "Framework"
    if len(name.split()) >= 2 and all(part[:1].isupper() for part in name.split() if part):
        return "Person"
    return "Product"


def add_entity(entities: List[Dict], name: str, entity_type: Optional[str] = None):
    name = normalize_space(name)
    if not name:
        return
    entity_type = entity_type or infer_type(name)
    if entity_type not in ALLOWED_TYPES:
        entity_type = infer_type(name)
    if not any(e["name"].lower() == name.lower() for e in entities):
        entities.append({"name": name, "type": entity_type})


def add_relationship(relationships: List[Dict], source: str, target: str, relation: str):
    source = normalize_space(source)
    target = normalize_space(target)
    relation = canonical_relation(relation)
    if not source or not target or source.lower() == target.lower():
        return
    if relation not in ALLOWED_RELATIONS:
        return
    item = {"source": source, "target": target, "relation": relation}
    if item not in relationships:
        relationships.append(item)


def extract_known_entities(text: str, entities: List[Dict]):
    for name, etype in KNOWN_ENTITIES.items():
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            add_entity(entities, name, etype)


def titlecase_candidate(s: str) -> str:
    s = normalize_space(s)
    return " ".join(w[:1].upper() + w[1:] if w else w for w in s.split())


def extract_capitalized_candidates(text: str, entities: List[Dict]):
    pattern = r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b"
    for m in re.finditer(pattern, text):
        candidate = normalize_space(m.group(1))
        if len(candidate) < 2:
            continue
        if candidate.lower() in {"the", "and", "but", "with"}:
            continue
        add_entity(entities, candidate, infer_type(candidate))


def relation_patterns() -> List[Tuple[str, str]]:
    return [
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+founded\s+([A-Z][A-Za-z0-9&.\-\s]+)", "FOUNDED"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+developed\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+created\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+authored\s+([A-Z][A-Za-z0-9&.\-\s]+)", "AUTHORED"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+hired\s+([A-Z][A-Za-z0-9&.\-\s]+)", "HIRED"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+integrates with\s+([A-Z][A-Za-z0-9&.\-\s]+)", "INTEGRATED_INTO"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+integrated into\s+([A-Z][A-Za-z0-9&.\-\s]+)", "INTEGRATED_INTO"),

        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was founded by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "FOUNDED_REV"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was developed by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED_REV"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was created by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "DEVELOPED_REV"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was authored by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "AUTHORED_REV"),
        (r"([A-Z][A-Za-z0-9&.\-\s]+?)\s+was hired by\s+([A-Z][A-Za-z0-9&.\-\s]+)", "HIRED_REV"),
    ]


def extract_relationships(text: str, entities: List[Dict], relationships: List[Dict]):
    for pattern, rel in relation_patterns():
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            a = normalize_space(m.group(1))
            b = normalize_space(m.group(2))
            if rel.endswith("_REV"):
                rel_name = rel.replace("_REV", "")
                source, target = b, a
            else:
                rel_name = rel
                source, target = a, b
            add_entity(entities, titlecase_candidate(source), infer_type(titlecase_candidate(source)))
            add_entity(entities, titlecase_candidate(target), infer_type(titlecase_candidate(target)))
            add_relationship(relationships, titlecase_candidate(source), titlecase_candidate(target), rel_name)


def build_indexes(relationships: List[Dict]):
    out_map = {}
    in_map = {}
    for r in relationships:
        out_map.setdefault(r["source"].lower(), []).append(r)
        in_map.setdefault(r["target"].lower(), []).append(r)
    return out_map, in_map


@app.get("/")
def root():
    return {"message": "GraphRAG API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-graph")
def extract_graph(payload: ExtractGraphRequest):
    text = payload.text
    entities: List[Dict] = []
    relationships: List[Dict] = []

    extract_known_entities(text, entities)
    extract_capitalized_candidates(text, entities)
    extract_relationships(text, entities, relationships)

    # Add implied entity types from relationships
    for r in relationships:
        add_entity(entities, r["source"], infer_type(r["source"]))
        add_entity(entities, r["target"], infer_type(r["target"]))

    return {
        "entities": entities,
        "relationships": relationships
    }


@app.post("/graph-query")
def graph_query(payload: GraphQueryRequest):
    question = payload.question.strip()
    relationships = payload.graph.get("relationships", [])
    out_map, in_map = build_indexes(relationships)
    q = question.lower()

    # Template 1: Who created/developed the framework that integrates with X?
    if "who" in q and ("integrates with" in q or "integrated into" in q):
        target_match = re.search(r"(?:integrates with|integrated into)\s+([A-Za-z0-9&.\-\s]+)\??", question, re.IGNORECASE)
        if target_match:
            target = normalize_space(target_match.group(1))
            framework = None
            for rel in relationships:
                if rel["relation"] == "INTEGRATED_INTO" and rel["target"].lower() == target.lower():
                    framework = rel["source"]
                    break
            if framework:
                for rel in relationships:
                    if rel["target"].lower() == framework.lower() and rel["relation"] in {"DEVELOPED", "AUTHORED", "FOUNDED", "HIRED"}:
                        return {
                            "answer": rel["source"],
                            "reasoning_path": [target, framework, rel["source"]],
                            "hops": 2
                        }

    # Template 2: Who founded/developed/authored X?
    m = re.search(r"who\s+(founded|developed|authored|hired)\s+([A-Za-z0-9&.\-\s]+)\??", question, re.IGNORECASE)
    if m:
        rel_word = m.group(1).upper()
        target = normalize_space(m.group(2))
        for rel in relationships:
            if rel["target"].lower() == target.lower() and rel["relation"] == rel_word:
                return {
                    "answer": rel["source"],
                    "reasoning_path": [target, rel["source"]],
                    "hops": 1
                }

    # Template 3: What integrates with X?
    m = re.search(r"what\s+(?:framework|product)?.*integrates with\s+([A-Za-z0-9&.\-\s]+)\??", question, re.IGNORECASE)
    if m:
        target = normalize_space(m.group(1))
        for rel in relationships:
            if rel["relation"] == "INTEGRATED_INTO" and rel["target"].lower() == target.lower():
                return {
                    "answer": rel["source"],
                    "reasoning_path": [target, rel["source"]],
                    "hops": 1
                }

    # Generic 2-hop fallback BFS
    names = set()
    for r in relationships:
        names.add(r["source"])
        names.add(r["target"])

    mentioned = [n for n in names if n.lower() in q]
    if mentioned:
        start = mentioned[0]
        # out edges
        for r1 in out_map.get(start.lower(), []):
            mid = r1["target"]
            for r2 in out_map.get(mid.lower(), []):
                return {
                    "answer": r2["target"],
                    "reasoning_path": [start, mid, r2["target"]],
                    "hops": 2
                }
            return {
                "answer": mid,
                "reasoning_path": [start, mid],
                "hops": 1
            }
        # incoming edges
        for r1 in in_map.get(start.lower(), []):
            mid = r1["source"]
            return {
                "answer": mid,
                "reasoning_path": [start, mid],
                "hops": 1
            }

    return {
        "answer": "Answer not found",
        "reasoning_path": [],
        "hops": 0
    }


@app.post("/community-summary")
def community_summary(payload: CommunitySummaryRequest):
    relationships = payload.relationships
    counts = {}

    for rel in relationships:
        counts[rel["source"]] = counts.get(rel["source"], 0) + 1
        counts[rel["target"]] = counts.get(rel["target"], 0) + 1

    center = max(counts, key=counts.get) if counts else (payload.entities[0] if payload.entities else "Unknown")

    phrases = []
    for rel in relationships:
        if rel["target"] == center and rel["relation"] == "FOUNDED":
            phrases.append(f"founded by {rel['source']}")
        elif rel["target"] == center and rel["relation"] == "DEVELOPED":
            phrases.append(f"developed by {rel['source']}")
        elif rel["target"] == center and rel["relation"] == "AUTHORED":
            phrases.append(f"authored by {rel['source']}")
        elif rel["target"] == center and rel["relation"] == "HIRED":
            phrases.append(f"hired by {rel['source']}")
        elif rel["source"] == center and rel["relation"] == "INTEGRATED_INTO":
            phrases.append(f"integrates with {rel['target']}")

    phrases = list(dict.fromkeys(phrases))
    if phrases:
        summary = f"This community centers around {center}, which is " + " and ".join(phrases) + "."
    else:
        summary = f"This community centers around {center} and its connected relationships."

    return {
        "community_id": payload.community_id,
        "summary": summary
    }