# Independent calculations for 14.3

The geometric identity is q·k = ||q||₂ ||k||₂ cos(theta). Here ||q||₂=√2. Both nonzero horizontal/vertical keys have cosine1/√2 with q, so theta=45°. Raw dot products are1 and100. Dividing q and each k by its own L2 length gives qhat=(1/√2,1/√2), khat1=(1,0), khat2=(0,1), and both dot products1/√2≈0.7071067812. Equal scores imply equal softmax weights1/2.

For scores(a,b), p(first)=1/(1+exp(b-a)); p(second)=1/(1+exp(a-b)). These are exact algebraic rearrangements using the maximum score to avoid needless overflow. Thus softmax(1,0)≈(0.7310585786,0.2689414214); softmax(1,100)≈(1.0112214926e-43,1); softmax(4,0)≈(0.9820137900,0.0179862100).

Replacing100 by0.1 gives raw dot products(1,0.1) and softmax≈(0.7109495026,0.2890504974). The positive vertical magnitude changes only length, so khat2 remains(0,1), and the normalized attention remains(0.5,0.5).

For a fixed positive multiplier alpha, softmax(alpha*s)=softmax(s/tau) with tau=1/alpha>0. The log ratio p_i/p_j is alpha*(s_i-s_j); larger alpha increases relative preference for a strictly higher score. Tied scores remain tied. For the example(1,0), tau2,1,0.25 yields first probabilities0.6224593312,0.7310585786,0.9820137900.

These calculations concern a local unscaled dot-product demonstration. Standard Transformer attention additionally uses1/√d_k (original §3.2.1), whereas the QKNorm paper uses cosine similarity and a learnable multiplier. The section presents its actual calculation and does not equate this toy forward pass with a whole trained model.

Units: angles in degrees, norms/scores/probabilities dimensionless in this abstract example. Softmax denominator is the sum over exactly two candidates per query. All candidate weights sum to1. Numbers displayed at four decimal places are checked to absolute5e-5; the independent float64 checks use atol1e-14, angles1e-12 degrees. Original float32's extremely small raw probability is a subnormal value and only claimed to be near0.
