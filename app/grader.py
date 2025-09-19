import re

# Heuristic weights
WEIGHTS = {
    "heading_levels": {  # explicit mapping for H1–H6chunk_content_heuristic
        1: 6,
        2: 5,
        3: 4,
        4: 3,
        5: 2,
        6: 1
    },
    "position": 3,   # max bonus for early chunks
    "bold": 3,
    "italic": 2,
    "code": 2,
    "list": 2
}

def score_md_chunk(chunk, pos, total_chunks, weights=WEIGHTS):
    score = 0

    # Heading score using explicit mapping
    heading_match = re.match(r"^(#+)\s", chunk)
    if heading_match:
        heading_level = len(heading_match.group(1))
        score += weights["heading_levels"].get(heading_level, 0)

    # Position score (early chunks get a bonus)
    score += max(0, weights["position"] - (pos / total_chunks) * weights["position"])

    # Bold and italic emphasis
    score += len(re.findall(r"\*\*(.*?)\*\*", chunk)) * weights["bold"]
    score += len(re.findall(r"\*(.*?)\*", chunk)) * weights["italic"]

    # Code blocks
    if chunk.strip().startswith("```"):
        score += weights["code"]

    # Lists
    if re.match(r"^[-*0-9]\s", chunk):
        score += weights["list"]

    return score