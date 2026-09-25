"""Parse a Markdown document into a hierarchical section tree.

This tree is the backbone for both retrieval strategies in the demo:
- vector_rag.py chunks the LEAVES of this tree and embeds them
- vectorless_rag.py lets Claude navigate the tree itself, the way
  PageIndex-style "vectorless RAG" tools do — titles and one-line
  summaries only, no embeddings anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Section:
    id: str
    title: str
    level: int
    content: str
    children: List["Section"] = field(default_factory=list)
    summary: Optional[str] = None

    def full_text(self) -> str:
        """This section's own text plus every descendant's — used once
        a node has been selected, to hand Claude the real content."""
        parts = [self.content]
        for child in self.children:
            parts.append(f"### {child.title}\n{child.full_text()}")
        return "\n\n".join(p.strip() for p in parts if p.strip())

    def leaves(self) -> List["Section"]:
        if not self.children:
            return [self]
        out: List[Section] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def flatten(self) -> List["Section"]:
        """Every node depth-first, self included — the full set of
        node IDs vectorless_rag.py can navigate to."""
        out = [self]
        for c in self.children:
            out.extend(c.flatten())
        return out


HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def parse_markdown(text: str, doc_id: str = "doc") -> Section:
    """Turn '# Title\\ncontent\\n## Sub\\ncontent' into a Section tree.
    Heading depth (# through ####) defines the hierarchy; everything
    else is content belonging to the nearest preceding heading.
    """
    root = Section(id=doc_id, title="ROOT", level=0, content="")
    path: Dict[int, Section] = {0: root}
    current_level = 0
    buffer: List[str] = []
    child_counts: Dict[Tuple[str, int], int] = {}

    def flush():
        nonlocal buffer
        if buffer:
            path[current_level].content += "\n".join(buffer).strip() + "\n\n"
            buffer = []

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()

            parent_level = level - 1
            while parent_level not in path:
                parent_level -= 1
            parent = path[parent_level]

            key = (parent.id, level)
            child_counts[key] = child_counts.get(key, 0) + 1
            node = Section(id=f"{parent.id}.{child_counts[key]}", title=title,
                            level=level, content="")
            parent.children.append(node)

            path[level] = node
            for lvl in [l for l in path if l > level]:
                del path[lvl]
            current_level = level
        else:
            buffer.append(line)
    flush()
    return root


def section_to_dict(node: Section) -> dict:
    return {
        "id": node.id, "title": node.title, "level": node.level,
        "content": node.content, "summary": node.summary,
        "children": [section_to_dict(c) for c in node.children],
    }


def section_from_dict(d: dict) -> Section:
    node = Section(id=d["id"], title=d["title"], level=d["level"],
                    content=d["content"], summary=d.get("summary"))
    node.children = [section_from_dict(c) for c in d["children"]]
    return node
