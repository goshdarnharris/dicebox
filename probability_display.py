from scipy.stats import binom


def compute_probability_display(disp_result, pivot):
    """
    Compute the probability display for the GUI.

    Shows the tail probability: "chance of being at least this lucky/unlucky".
    Green for above expected, red for below expected.

    Args:
        disp_result: list of 6 ints — counts of each face value [#1s, #2s, #3s, #4s, #5s, #6s]
        pivot: int 1-5 — dividing line between low and high

    Returns:
        (text, font_size, color_hex) or None if there's nothing to display
    """
    n_dice = sum(disp_result)
    if n_dice < 1:
        return None

    high_sum = sum(disp_result[pivot:])
    prob_high = (6 - pivot) / 6.0
    expected = n_dice * prob_high

    if high_sum >= expected:
        # Lucky: P(X >= high_sum)
        tail_prob = binom.sf(high_sum - 1, n_dice, prob_high)  # sf(k-1) = P(X >= k)
        lucky = True
    else:
        # Unlucky: P(X <= high_sum)
        tail_prob = binom.cdf(high_sum, n_dice, prob_high)
        lucky = False

    tail_prob = max(0.0001, min(1.0, tail_prob))

    # Format as percentage
    if tail_prob >= 0.1:
        text = f"{tail_prob * 100:.1f}%"
    elif tail_prob >= 0.01:
        text = f"{tail_prob * 100:.2f}%"
    else:
        text = f"{tail_prob * 100:.3f}%"

    # Color intensity based on how extreme the probability is.
    # 50% = white (neutral), approaching 0% = fully saturated.
    # Use log scale so small probabilities stand out.
    # -log2(0.5) = 1, -log2(0.01) ~= 6.6
    import math
    intensity = min(1.0, -math.log2(max(tail_prob, 0.0001)) / 6.0)

    if lucky:
        r, g, b = int(255 * (1 - intensity)), 255, int(255 * (1 - intensity))
    else:
        r, g, b = 255, int(255 * (1 - intensity)), int(255 * (1 - intensity))
    color = f"#{r:02x}{g:02x}{b:02x}"

    # Font size: base 22, grows with extremity
    font_size = int(22 + intensity * 30)

    return text, font_size, color
