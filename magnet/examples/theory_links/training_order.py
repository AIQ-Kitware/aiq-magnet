"""
Practice establishes a phenomenon, and theory is asked to explain it.

An online learner sees the same four observations twice, in opposite orders,
and ends up classifying a probe point differently. Nothing is random and
nothing is truncated early: the updates are the textbook perceptron rule, run
once over the data in each order.

The experiment claims only that this happens. Why it happens, and when a
learning rule should be order-invariant, is the question it points at.
"""
import magnet.theory as theory

#: Four points in the plane with labels, hand-picked so one pass in either
#: direction leaves the boundary in a different place. Two of them are on the
#: same side of any separating line, which is what leaves the last update in
#: charge of the outcome.
OBSERVATIONS = (
    ((2.0, 1.0), 1),
    ((1.0, 2.0), 1),
    ((-1.0, 0.5), -1),
    ((0.5, -1.0), -1),
)

#: Between the two learned boundaries, where the runs disagree.
PROBE = (-0.8, 0.8)


def train_online(observations, learning_rate: float = 1.0) -> tuple:
    """
    One pass of the perceptron rule over the observations, in order.

    Args:
        observations: pairs of ``((x, y), label)`` with labels in ``{-1, 1}``.
        learning_rate (float): step size.

    Returns:
        tuple: the weights ``(w_x, w_y, bias)``.

    Example:
        >>> from magnet.examples.theory_links.training_order import (
        ...     OBSERVATIONS, train_online)
        >>> train_online(OBSERVATIONS)
        (1.5, 2.0, 0.0)
        >>> train_online(tuple(reversed(OBSERVATIONS)))
        (1.5, 2.5, -1.0)
    """
    w_x = w_y = bias = 0.0
    for (x, y), label in observations:
        score = w_x * x + w_y * y + bias
        if label * score <= 0:
            w_x += learning_rate * label * x
            w_y += learning_rate * label * y
            bias += learning_rate * label
    return (w_x, w_y, bias)


def predict(weights, point) -> int:
    """
    Which side of the learned boundary a point falls on.

    Example:
        >>> from magnet.examples.theory_links.training_order import predict
        >>> predict((1.0, 1.0, 0.0), (1.0, 1.0))
        1
    """
    w_x, w_y, bias = weights
    x, y = point
    return 1 if w_x * x + w_y * y + bias > 0 else -1


@theory.motivates('Examples.TrainingOrder.Why')
def training_order_sensitivity(observations=OBSERVATIONS, probe=PROBE) -> dict:
    """
    Train on the same observations forwards and backwards, and compare.

    Returns:
        dict: the two weight vectors, the two probe predictions, and whether
            they disagree.

    Example:
        >>> from magnet.examples.theory_links.training_order import (
        ...     training_order_sensitivity)
        >>> result = training_order_sensitivity()
        >>> result['disagrees']
        True
        >>> result['forward_prediction'], result['reverse_prediction']
        (1, -1)
    """
    forward = train_online(observations)
    reverse = train_online(tuple(reversed(observations)))
    forward_prediction = predict(forward, probe)
    reverse_prediction = predict(reverse, probe)
    return {
        'forward_weights': forward,
        'reverse_weights': reverse,
        'forward_prediction': forward_prediction,
        'reverse_prediction': reverse_prediction,
        'disagrees': forward_prediction != reverse_prediction,
    }
