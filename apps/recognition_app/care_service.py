"""Curated, offline care guidance; never generated from an LLM or a live web search."""
from copy import deepcopy
import json
from pathlib import Path

CATALOG = json.loads((Path(__file__).parent / "knowledge_base/care_guides.json").read_text())


def care_response(label, uncertain=False):
    unconfirmed = bool(uncertain) or label not in CATALOG["guides"]
    guide = deepcopy(CATALOG["unconfirmed"] if unconfirmed else CATALOG["guides"][label])
    guide["version"] = CATALOG["version"]
    guide["checked_at"] = CATALOG["checked_at"]
    guide["status"] = "unconfirmed" if unconfirmed else "reference"
    guide["sources"] = [
        {**CATALOG["sources"][source_id], "id": source_id, "section": section}
        for source_id, section in guide.pop("source_sections").items()
    ]
    return {
        "guidance": guide["summary"],
        "care_guide": guide,
        "speech_guidance": guide["summary"] + " " + guide["caution"],
    }
