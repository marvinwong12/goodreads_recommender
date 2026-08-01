import numpy as np

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Safely min-max normalizes scores to [0, 1] while ignoring -inf/nan values."""
    finite_mask = np.isfinite(scores)
    
    if not np.any(finite_mask):
        return np.zeros_like(scores)
        
    min_val = np.min(scores[finite_mask])
    max_val = np.max(scores[finite_mask])
    
    denom = max_val - min_val
    if denom == 0:
        # Avoid division by zero if all scores are identical
        scaled = np.zeros_like(scores)
        scaled[~finite_mask] = -np.inf
        return scaled
        
    scaled = np.full_like(scores, -np.inf, dtype=np.float64)
    scaled[finite_mask] = (scores[finite_mask] - min_val) / denom
    return scaled

def recency_weighted_profile(embeddings, ordered_book_indices, decay_rate=0.9):
    """
    Builds a user's semantic profile as an exponentially recency-weighted
    average of their read books' embeddings, instead of a plain mean. A flat
    mean dilutes short-term taste shifts (e.g. a genre binge, moving on to a
    new series) into an all-time average; weighting recent reads more heavily
    keeps the profile responsive to what the user is currently into.

    Args:
        embeddings (np.ndarray): Book embedding matrix (num_books, dim).
        ordered_book_indices (list[int]): The user's read book indices,
            ordered oldest -> most recent.
        decay_rate (float): Weight ratio between consecutive reads; the most
            recent read gets weight 1, each earlier one decays by this factor.

    Returns:
        profile (np.ndarray): L2-normalized weighted profile vector.
    """
    n = len(ordered_book_indices)
    weights = decay_rate ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()
    profile = (embeddings[ordered_book_indices] * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile


def fuse_scores(lightgcn_scores, semantic_scores, user_interaction_count, decay_rate=0.05, min_alpha=0.15):
    """
    Blends collaborative and semantic scores dynamically.
    
    Args:
        lightgcn_scores (list/np.array): Raw dot product scores from LightGCN.
        semantic_scores (list/np.array): Raw similarity scores from Vector DB.
        user_interaction_count (int): How many books the user has previously read.
        decay_rate (float): Lambda controlling how fast alpha drops as read count grows.
        min_alpha (float): The absolute minimum weight semantic search will ever get.
        
    Returns:
        final_scores (np.array): The blended, final scores used for sorting.
        alpha (float): The actual semantic weight calculated for this user.
    """
    # 1. Calculate dynamic alpha for this specific user
    # Formula: max(min_alpha, e^(-decay * interactions))
    raw_alpha = np.exp(-decay_rate * user_interaction_count)
    alpha = max(min_alpha, raw_alpha)
    
    # 2. Normalize both sets of scores independently
    lightgcn_norm = min_max_normalize(lightgcn_scores)
    semantic_norm = min_max_normalize(semantic_scores)
    
    # 3. Apply the fusion weighting
    final_scores = (alpha * semantic_norm) + ((1.0 - alpha) * lightgcn_norm)
    
    return final_scores, alpha

if __name__ == "__main__":
    # Simulated scores for 5 candidate books retrieved for a user
    simulated_lightgcn = np.array([12.4, 8.1, -1.2, 14.5, 3.3])
    simulated_semantic = np.array([0.92, 0.88, 0.95, 0.70, 0.81])

    # Scenario A: A Brand New User (Only 1 book read)
    scores_new_user, alpha_new = fuse_scores(
        lightgcn_scores=simulated_lightgcn, 
        semantic_scores=simulated_semantic, 
        user_interaction_count=1
    )

    # Scenario B: A Power User (45 books read)
    scores_power_user, alpha_power = fuse_scores(
        lightgcn_scores=simulated_lightgcn, 
        semantic_scores=simulated_semantic, 
        user_interaction_count=45
    )

    print(f"--- Cold Start User (Read 1 book) ---")
    print(f"Alpha (Semantic Weight) : {alpha_new:.2f}")
    print(f"Final Candidate Scores  : {np.round(scores_new_user, 3)}\n")

    print(f"--- Power User (Read 45 books) ---")
    print(f"Alpha (Semantic Weight) : {alpha_power:.2f}")
    print(f"Final Candidate Scores  : {np.round(scores_power_user, 3)}")