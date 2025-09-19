import re
import numpy as np

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


def symmetric_overlap_func(target_vector, candidate_vector):
    """
    Compute the symmetric overlap similarity between two sets of vectors.

    Args:
        target_vector (list or np.ndarray): list of N vectors representing the target document/chunks.
        candidate_vector (list or np.ndarray): list of M vectors representing the candidate document/chunks.

    Returns:
        float: symmetric overlap score between 0 and 1, higher means more similar.

    Explanation:
        1. Converts the input lists to NumPy arrays for efficient computation.
        2. Normalizes all vectors to unit length so that dot product equals cosine similarity.
        3. Computes the full cosine similarity matrix of shape [N x M], where each entry [i, j] is 
        the similarity between target chunk i and candidate chunk j.
        4. For each target chunk, finds the maximum similarity with any candidate chunk.
        5. For each candidate chunk, finds the maximum similarity with any target chunk.
        6. Averages these maximum similarities in both directions (target → candidate and 
        candidate → target) and takes 0.5 * sum to get a symmetric overlap score.
        - Symmetric overlap is important because it ensures that both sets “match” each other,
            avoiding bias if one set is larger or has unmatched chunks.
        7. Returns a single float representing overall similarity.

    Note:
        - This method naturally handles different numbers of chunks in the target and candidate.
        - Symmetric overlap is widely used in semantic search and multi-chunk embedding comparisons
        to give a fair and balanced similarity measure.
    """

    
    
    target_matrix = np.array(target_vector)       # shape: [N, dim]
    candidate_matrix = np.array(candidate_vector) # shape: [M, dim]

    # normalize vectors
    target_norm = target_matrix / np.linalg.norm(target_matrix, axis=1, keepdims=True)
    candidate_norm = candidate_matrix / np.linalg.norm(candidate_matrix, axis=1, keepdims=True)

    # cosine similarity matrix: [N x M]
    cos_sim_matrix = target_norm @ candidate_norm.T

    # compute symmetric overlap
    max_target_to_candidate = np.max(cos_sim_matrix, axis=1)  # length N
    max_candidate_to_target = np.max(cos_sim_matrix, axis=0)  # length M

    symmetric_overlap = 0.5 * (np.mean(max_target_to_candidate) + np.mean(max_candidate_to_target))
    return symmetric_overlap
