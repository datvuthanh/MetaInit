# MetaInit

MetaInit is based on a hypothesis that good initializations make gradient descent easier by starting in regions that look locally linear with minimal second order effects. We formalize this notion via a quantity that we call the gradient quotient, which can be computed with any architecture or dataset. MetaInit minimizes this quantity efficiently by using gradient descent to tune the norms of the initial weight matrices.

This is a sample of code MetaInit in Tensorflow version.
