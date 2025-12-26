"""
Temporal scoring functions for LAR-RAG.

Implements RCA-aware temporal decay that penalizes future logs.
"""

import math


def g_time(delta_t_s: float, lambda_pre: float, lambda_post: float) -> float:
    """
    Asymmetric temporal proximity kernel for RCA.
    
    Root-cause logs typically occur BEFORE incidents.
    Logs AFTER incidents are less relevant and penalized more heavily.
    
    Args:
        delta_t_s: Time difference in seconds (incident_time - log_time)
                  Positive = log before incident (expected)
                  Negative = log after incident (unusual)
        lambda_pre: Decay rate for logs before incident (smaller = slower decay)
        lambda_post: Decay rate for logs after incident (larger = faster decay)
        
    Returns:
        Temporal proximity score in [0, 1]
        
    Examples:
        >>> g_time(60, 0.001, 0.01)  # Log 60s before incident
        0.9418
        >>> g_time(-60, 0.001, 0.01)  # Log 60s after incident
        0.5488  # Stronger penalty
    """
    if delta_t_s >= 0:
        # Log before incident: normal decay
        return math.exp(-lambda_pre * delta_t_s)
    else:
        # Log after incident: stronger penalty
        return math.exp(-lambda_post * abs(delta_t_s))


def temporal_score_symmetric(delta_t_s: float, lambda_decay: float) -> float:
    """
    Symmetric temporal decay (baseline for ablation studies).
    
    Args:
        delta_t_s: Absolute time difference in seconds
        lambda_decay: Decay rate
        
    Returns:
        Temporal proximity score in [0, 1]
    """
    return math.exp(-lambda_decay * abs(delta_t_s))


def temporal_score_windowed(delta_t_s: float, window_seconds: float) -> float:
    """
    Hard window temporal scoring (baseline for ablation studies).
    
    Args:
        delta_t_s: Time difference in seconds
        window_seconds: Window size (W)
        
    Returns:
        1.0 if within window, 0.0 otherwise
    """
    return 1.0 if abs(delta_t_s) <= window_seconds else 0.0


if __name__ == "__main__":
    # Test asymmetric decay
    print("Asymmetric Temporal Decay (RCA-aware):")
    print("-" * 60)
    
    lambda_pre = 0.001  # Slow decay for past logs
    lambda_post = 0.01   # Fast decay for future logs
    
    test_deltas = [-300, -60, -10, 0, 10, 60, 300, 600]
    
    for delta in test_deltas:
        score = g_time(delta, lambda_pre, lambda_post)
        direction = "BEFORE" if delta >= 0 else "AFTER "
        print(f"Δt = {delta:4d}s ({direction} incident): score = {score:.4f}")
    
    print("\n" + "=" * 60)
    print("Notice: Logs AFTER incident decay faster (lower scores)")
