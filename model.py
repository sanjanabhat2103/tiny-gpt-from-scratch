"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    return {char: i for i, char in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    return {i: char for i, char in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    return "".join(decode_int(i, itos) for i in ids)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    return np.asarray(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    return arr[: , j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    return arr[r0: r1, c0: c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, axis = 0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    return np.sum(arr, axis = 1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis = axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    return np.sum(arr, axis = axis, keepdims = True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    exp = array_exp(logits)
    s = sum_all(exp)
    return exp / s

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    value = array_exp(large_value)
    return {"naive_exp": float(value), "overflowed": bool(np.isinf(value))}

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    exp = array_exp(logits - max_along_axis(logits, axis = 0))
    s = sum_all(exp)
    return exp / s

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    row_max = max_along_axis(logits, axis = 1)[: , None]
    shifted = logits - row_max
    exp_values = array_exp(shifted)
    row_sums = sum_keepdims(exp_values, axis = 1)
    return exp_values / row_sums

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    if not text_blob:
        raise ValueError("Input must not be empty.")
    if not isinstance(text_blob, str):
        raise TypeError("Input must be a string.")
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    return np.array(encode_string(text, stoi), dtype = np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    idx = int(n * train_frac)
    return idx

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    return (data[: split_idx], data[split_idx: ])

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    if default_size < 1:
        return 1
    return int(default_size)

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    return data[i: i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    return data[i + 1: block_size + i + 1]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    max_offset = data_len - block_size - 1
    return rng.integers(0, max_offset + 1, size = batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    return np.stack([data[offset: offset + block_size] for offset in offsets])

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    return np.stack([data[offset + 1: offset + block_size + 1] for offset in offsets])

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    offsets = sample_random_batch_offsets(data.size, block_size, batch_size, rng)
    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)
    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.zeros((vocab_size, vocab_size), dtype = int)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for i in range(len(data) - 1):
        curr = data[i]
        next = data[i + 1]
        n_matrix[curr, next] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    counts = allocate_count_matrix(vocab_size)
    np.add.at(counts, (data[: -1], data[1: ]), 1)
    return counts

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    return sum_keepdims(n_matrix, axis = 1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    probs = p_matrix[current_id]
    return int(rng.choice(len(probs), p = probs))

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    sequence = np.empty(length, dtype = np.int64)
    sequence[0] = start_id
    for i in range(1, length):
        sequence[i] = sample_next_token(p_matrix, sequence[i - 1], rng)
    return sequence

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    return decode_ids(ids, itos)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    return np.log(p_matrix[current_id, next_id])

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    total = 0
    for i in range(len(data) - 1):
        total -= log_prob_of_pair(p_matrix, data[i], data[i + 1])
    return total

# Step 56 - average_nll
def average_nll(p_matrix, data):
    return sum_negative_log_probs(p_matrix, data) / (len(data) - 1)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    one_hot = make_2d_zeros(len(ids), vocab_size)
    one_hot[np.arange(len(ids)), ids] = 1
    return one_hot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    one_hot = one_hot_encode_batch(ids, w.shape[0])
    return {'onehot_result': one_hot @ w, 'index_result': w[ids]}

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    return probs[np.arange(len(targets)), targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    correct_probs = gather_correct_token_probs(probs, targets)
    arr_log = array_log(correct_probs)
    return -np.mean(arr_log)

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    return (
        'For softmax cross-entropy, the gradient dL/dlogits is '
        '(probs - onehot(targets)) / B. '
        'This follows from combining the softmax derivative with the '
        'negative log-likelihood gradient.'
    )

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    dlogits = probs.copy()
    dlogits[np.arange(len(targets)), targets] -= 1
    return dlogits / len(targets)

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros((vocab_size, dlogits.shape[1]))
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    return np.asarray(w - learning_rate * dw)

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, w.shape[0])
    new_w = w - learning_rate * dw
    return {'w': new_w, 'loss': float(loss)}

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    rng = np.random.default_rng(0)
    loss_history = []
    for step in range(num_steps):
        X, Y = get_batch(data, block_size, batch_size, rng)
        ids = X.reshape(-1)
        targets = Y.reshape(-1)
        result = run_one_training_step(w, ids, targets, learning_rate)
        w = result["w"]
        if step % log_every == 0:
            loss_history.append(result["loss"])
    return {"w": w, "loss_history": loss_history}

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    ids = [start_id]
    current_id = start_id
    for _ in range(num_tokens):
        probs = logits_to_probs_rowwise(w[current_id][None, : ])[0]
        current_id = np.random.choice(len(probs), p = probs)
        ids.append(current_id)
    return decode_ids(ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    # TODO: compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.
    y = x @ w 
    return {'y': y, 'cache': {'x': x, 'w': w}}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    return (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    return (
        "Forward: Y = X @ W.\n"
        "Shapes: X (B, D_in), W (D_in, D_out), dY (B, D_out).\n"
        "By the chain rule, dL/dW = X.T @ dY.\n"
        "The result has shape (D_in, D_out), matching W."
    )

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    w = cache['w']
    return dy @ w.T

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    x = cache['x']
    return x.T @ dy

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    y = x + b 
    return {'y': y, 'cache': {'b_shape': b.shape}}

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    return dy.sum(axis = 0).reshape(cache["b_shape"])

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    y = np.maximum(0.0, x)
    return {"y": y, "cache": {"x": x}}

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    x = cache["x"]
    dx = dy * (x > 0)
    return dx

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    batch_size = probs.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), targets] -= 1.0
    return dlogits / batch_size

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    return sum_keepdims(x, axis = -1) / x.shape[-1]

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    var = np.mean((x - mean) ** 2, axis = -1, keepdims = True)
    return var

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    return (x - mean) / (np.sqrt(var + eps))

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)
    scaled = elementwise_multiply(x_hat, gamma)
    y = vector_matrix_broadcast_add(scaled, beta)
    cache = {
        "x": x,
        "x_hat": x_hat,
        "mean": mean,
        "var": var,
        "gamma": gamma,
        "eps": eps,
    }
    return {"y": y, "cache": cache}

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    D = dy.shape[1]
    dx = dy - np.sum(dy, axis = 1, keepdims = True) / D
    return dx

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    var = cache["var"]
    eps = cache["eps"]
    std = np.sqrt(var + eps)
    dx_centered = dy / std
    return dx_centered

# Step 90 - layernorm_backward_full
def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    x = cache["x"]
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]
    B, D = x.shape
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis = 0)
    dx_hat = dy * gamma
    x_centered = x - mean
    inv_std = 1.0 / np.sqrt(var + eps)
    dx = (inv_std / D) * (
        D * dx_hat
        - np.sum(dx_hat, axis = 1, keepdims = True)
        - x_centered * (inv_std ** 2)
          * np.sum(dx_hat * x_centered, axis = 1, keepdims = True)
    )
    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta,
    }

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    """Return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm."""
    x = cache["x"]
    x_hat = cache["x_hat"]
    mean = cache["mean"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]
    D = x.shape[1]
    dgamma = np.sum(d_out * x_hat, axis = 0)
    dbeta = np.sum(d_out, axis = 0)
    dx_hat = d_out * gamma
    x_centered = x - mean
    inv_std = 1.0 / np.sqrt(var + eps)
    dx = (inv_std / D) * (
        D * dx_hat
        - np.sum(dx_hat, axis = 1, keepdims = True)
        - x_centered * (inv_std ** 2)
        * np.sum(dx_hat * x_centered, axis = 1, keepdims = True)
    )
    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta,
    }

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    return np.random.randn(vocab_size, d_model) * scale

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    out = embedding_matrix[token_ids]
    cache = {
        "token_ids": token_ids,
        "vocab_size": embedding_matrix.shape[0],
    }
    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    token_ids = cache["token_ids"]
    vocab_size = cache["vocab_size"]
    d_model = d_out.shape[-1]
    dE = np.zeros((vocab_size, d_model), dtype = d_out.dtype)
    np.add.at(dE, token_ids, d_out)
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    P = make_2d_random(block_size, d_model, seed = None)
    return scale_w_small(P, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    return positional_matrix[0: seq_len, : ]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    return token_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    d_token_emb = d_out
    d_pos_emb = sum_axis0(d_out)
    return {
        "d_token_emb": d_token_emb,
        "d_pos_emb": d_pos_emb,
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    return {
        "Wq": scale_w_small(make_2d_random(d_model, d_head, seed = 0), scale),
        "Wk": scale_w_small(make_2d_random(d_model, d_head, seed = 1), scale),
        "Wv": scale_w_small(make_2d_random(d_model, d_head, seed = 2), scale),
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    return x @ w_q

# Step 101 - compute_key
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    return x @ w_k

# Step 102 - compute_value
def compute_value(x, w_v):
    return x @ w_v

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    return q @ np.swapaxes(k, -1, -2)

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype = bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    max_scores = np.max(masked_scores, axis = -1, keepdims = True)
    exp_scores = np.exp(masked_scores - max_scores)
    return exp_scores / np.sum(exp_scores, axis = -1, keepdims = True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    attn_out = cache["attn_out"]
    w_o = cache["w_o"]
    d_attn_out = d_proj @ w_o.T
    dw_o = np.einsum("btd,btm->dm", attn_out, d_proj)
    return {
        "d_attn_out": d_attn_out,
        "dw_o": dw_o,
    }

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V."""
    attn = cache["attn"]
    v = cache["v"]
    d_attn = d_attn_out @ np.swapaxes(v, -1, -2)
    d_v = np.swapaxes(attn, -1, -2) @ d_attn_out
    return {
        "d_attn": d_attn,
        "d_v": d_v,
    }

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    attn = cache["attn"]
    causal_mask = cache["causal_mask"]
    dot = np.sum(d_attn * attn, axis = -1, keepdims = True)
    d_masked_scores = attn * (d_attn - dot)
    d_masked_scores = np.where(causal_mask, d_masked_scores, 0.0)
    return d_masked_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    q = cache["q"]
    k = cache["k"]
    d_q = d_scores @ k
    d_k = np.swapaxes(d_scores, -1, -2) @ q
    return {
        "d_q": d_q,
        "d_k": d_k,
    }

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    x = cache["x"]
    w_q = cache["w_q"]
    w_k = cache["w_k"]
    w_v = cache["w_v"]
    dx_q = d_q @ w_q.T
    dx_k = d_k @ w_k.T
    dx_v = d_v @ w_v.T
    dw_q = np.einsum("btd,bth->dh", x, d_q)
    dw_k = np.einsum("btd,bth->dh", x, d_k)
    dw_v = np.einsum("btd,bth->dh", x, d_v)
    dx = dx_q + dx_k + dx_v
    return {
        "dx": dx,
        "dw_q": dw_q,
        "dw_k": dw_k,
        "dw_v": dw_v,
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    d_head = d_model // n_heads
    return {
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    return {
        "Wq": scale_w_small(make_2d_random(d_model, d_model, seed = 0), scale),
        "Wk": scale_w_small(make_2d_random(d_model, d_model, seed = 1), scale),
        "Wv": scale_w_small(make_2d_random(d_model, d_model, seed = 2), scale),
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    return scale_w_small(
        make_2d_random(d_model, d_model, seed = 0),
        scale
    )

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    B, T, d_model = x.shape
    return x.reshape(B, T, n_heads, d_head)

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    return np.swapaxes(x_heads, 1, 2)

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    return config["n_heads"]

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    shape = get_array_shape(x)
    return shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError("n_heads must evenely divide d_model")
    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    B, n_heads, T, _ = scores.shape
    masked_scores = apply_causal_mask(scores, mask)
    flat_scores = masked_scores.reshape(B * n_heads * T, T)
    weights = stable_softmax_2d_rowwise(flat_scores)
    return weights.reshape(B, n_heads, T, T)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    return weights @ v_heads

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x_heads):
    return np.transpose(x_heads, (0, 2, 1, 3))

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    return x_heads_back.shape[1]

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    B = x_heads_back.shape[0]
    T = x_heads_back.shape[1]
    n_heads = x_heads_back.shape[2]
    d_head = x_heads_back.shape[3]
    d_model = n_heads * d_head
    return x_heads_back.reshape(B, T, d_model)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    linear_result = linear_forward(merged, w_out)
    bias_result = bias_add_forward(linear_result["y"], b_out)
    return {
        "out": bias_result["y"],
        "cache": {
            "merged": merged,
            "w_out": w_out,
        },
    }

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    B = shape_info["B"]
    T = shape_info["T"]
    n_heads = shape_info["n_heads"]
    d_head = shape_info["d_head"]
    d_transposed = d_merged.reshape(B, T, n_heads, d_head)
    return transpose_heads_to_front(d_transposed)

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    linear_result = linear_forward(x, w1)
    bias_result = bias_add_forward(linear_result["y"], b1)
    return {
        "h1": bias_result["y"],
        "cache": {
            "x": x,
            "w1": w1,
        },
    }

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    result = relu_forward(h1)
    a1 = result["y"]
    cache = {
        "h1": h1,
    }
    return a1, cache

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    """Project a1 (B, T, d_ff) down to (B, T, d_model) using w2 and b2."""
    linear_result = linear_forward(a1, w2)
    bias_result = bias_add_forward(linear_result["y"], b2)
    return {
        "h2": bias_result["y"],
        "cache": {
            "a1": a1,
            "w2": w2,
        },
    }

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN."""
    x = cache["x"]
    w1 = cache["w1"]
    h1 = cache["h1"]
    a1 = cache["a1"]
    w2 = cache["w2"]
    B, T, d_model = x.shape
    d_ff = w1.shape[1]
    d_out_2d = d_out.reshape(B * T, d_model)
    a1_2d = a1.reshape(B * T, d_ff)
    h1_2d = h1.reshape(B * T, d_ff)
    x_2d = x.reshape(B * T, d_model)
    linear2_cache = {"x": a1_2d, "w": w2}
    da1_2d = linear_backward_dx(d_out_2d, linear2_cache)
    dw2 = linear_backward_dw(d_out_2d, linear2_cache)
    db2 = bias_add_backward_db(
        d_out_2d,
        {"b_shape": (d_model,)}
    )
    dh1_2d = relu_backward(da1_2d, {"x": h1_2d})
    linear1_cache = {"x": x_2d, "w": w1}
    dx_2d = linear_backward_dx(dh1_2d, linear1_cache)
    dw1 = linear_backward_dw(dh1_2d, linear1_cache)
    db1 = bias_add_backward_db(
        dh1_2d,
        {"b_shape": (d_ff,)}
    )
    return {
        "dx": dx_2d.reshape(B, T, d_model),
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    return x + sublayer_out

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    return d_y.copy(), d_y.copy()

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params, eps=1e-5):
    ln_result = layernorm_forward_affine(
        x,
        ln_params["gamma"],
        ln_params["beta"],
        eps
    )
    sublayer_result = sublayer_fn(
        ln_result["y"],
        sublayer_params
    )
    residual_result = residual_forward(
        x,
        sublayer_result["y"]
    )
    return {
        "y": residual_result,
        "cache": {
            "x": x,
            "ln_cache": ln_result["cache"],
            "sublayer_cache": sublayer_result["cache"],
        },
    }

# Step 138 - transformer_block_forward
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward."""
    ln1 = layernorm_forward_affine(
        x,
        block_params["ln1"]["gamma"],
        block_params["ln1"]["beta"],
        1e-5,
    )
    attn = block_params["attn"]
    n_heads = attn["n_heads"]
    d_head = compute_d_head(x.shape[-1], n_heads)
    q = compute_query(ln1["y"], attn["Wq"])
    k = compute_key(ln1["y"], attn["Wk"])
    v = compute_value(ln1["y"], attn["Wv"])
    q = transpose_heads_to_front(reshape_to_heads(q, n_heads, d_head))
    k = transpose_heads_to_front(reshape_to_heads(k, n_heads, d_head))
    v = transpose_heads_to_front(reshape_to_heads(v, n_heads, d_head))
    scores = scale_attention_scores(
        q @ np.swapaxes(k, -1, -2), d_head
    )
    weights = multihead_masked_softmax_scores(
        scores, build_causal_mask(x.shape[1])
    )
    attn_out = multihead_weighted_sum(weights, v)
    attn_out = merge_heads_to_d_model(
        transpose_heads_to_back(attn_out)
    )
    attn_out = apply_output_projection(attn_out, attn["Wo"])
    attn_out = attn_out + attn["bo"]
    x2 = x + attn_out
    ln2 = layernorm_forward_affine(
        x2,
        block_params["ln2"]["gamma"],
        block_params["ln2"]["beta"],
        1e-5,
    )
    ffn = block_params["ffn"]
    h1 = ffn_linear_one_forward(
        ln2["y"], ffn["w1"], ffn["b1"]
    )
    a1 = ffn_activation_forward(h1["h1"])[0]
    h2 = ffn_linear_two_forward(
        a1, ffn["w2"], ffn["b2"]
    )
    return {
        "y": x2 + h2["h2"],
        "cache": {
            "attn_branch": ln1["cache"],
            "ffn_branch": ln2["cache"],
        },
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block."""
    x = cache["attn_branch"]["x"]
    full_cache = _complete_block_cache(x, block_params)
    ffn_branch = full_cache["ffn_branch"]
    d_ffn_z, ffn_grads = _ffn_sublayer_backward(
        d_y,
        ffn_branch["sublayer_cache"],
        block_params["ffn"],
    )
    d_h1_ln, d_ln2_gamma, d_ln2_beta = layernorm_backward_affine(
        d_ffn_z,
        ffn_branch["ln_cache"],
    )
    d_h1 = d_y + d_h1_ln
    attn_branch = full_cache["attn_branch"]
    d_attn_z, attn_grads = _attn_sublayer_backward(
        d_h1,
        attn_branch["sublayer_cache"],
        block_params["attn"],
    )
    d_x_ln, d_ln1_gamma, d_ln1_beta = layernorm_backward_affine(
        d_attn_z,
        attn_branch["ln_cache"],
    )
    d_x = d_h1 + d_x_ln
    grads = {
        "ln1": {
            "gamma": d_ln1_gamma,
            "beta": d_ln1_beta,
        },
        "ln2": {
            "gamma": d_ln2_gamma,
            "beta": d_ln2_beta,
        },
        "attn": attn_grads,
        "ffn": ffn_grads,
    }
    return d_x, grads

# Step 140 - stack_transformer_blocks
def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    blocks = []
    d_head = d_model // n_heads
    for i in range(n_layers):
        block = {
            'ln1': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'attn': {
                'Wq': scale_w_small(make_2d_random(d_model, d_model, seed = 0), 0.02),
                'Wk': scale_w_small(make_2d_random(d_model, d_model, seed = 1), 0.02),
                'Wv': scale_w_small(make_2d_random(d_model, d_model, seed = 2), 0.02),
                'Wo': scale_w_small(make_2d_random(d_model, d_model, seed = 3), 0.02),
                'bo': np.zeros(d_model)
            },
            'ln2': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'ffn': {
                'W1': scale_w_small(make_2d_random(d_model, d_ff, seed = 4), 0.02),
                'b1': np.zeros(d_ff),
                'W2': scale_w_small(make_2d_random(d_ff, d_model, seed = 5), 0.02),
                'b2': np.zeros(d_model)
            }
        }
        blocks.append(block)
    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    caches = []
    h = x
    for block in blocks:
        out = transformer_block_forward(h, block)
        h = out["y"]
        caches.append(out["cache"])
    return h, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y      : (B, T, d_model) upstream gradient at the top of the stack
      caches   : list of per-block forward caches
      blocks   : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    grads_list = []
    d_x = d_y
    for block, cache in zip(reversed(blocks), reversed(caches)):
        d_x, grads = transformer_block_backward(d_x, cache, block)
        grads_list.append(grads)
    grads_list.reverse()
    return d_x, grads_list

# Step 143 - final_layernorm_forward
import numpy as np

def final_layernorm_forward(x, gamma, beta, eps=1e-5):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    B, T, d_model = x.shape
    x_flat = x.reshape(-1, d_model)
    mean = np.mean(x_flat, axis = 1, keepdims = True)
    var = np.var(x_flat, axis = 1, keepdims = True)
    x_hat = (x_flat - mean) / np.sqrt(var + eps)
    y_flat = gamma * x_hat + beta
    y = y_flat.reshape(B, T, d_model)
    cache = {
        'x': x,
        'mean': mean.reshape(B, T, 1),
        'var': var.reshape(B, T, 1),
        'x_hat': x_hat.reshape(B, T, d_model),
        'gamma': gamma
    }
    return y, cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    lin_out = linear_forward(x, w_lm)
    y = lin_out['y']
    bias_out = bias_add_forward(y, b_lm)
    logits = bias_out['y']
    return {
        'logits': logits,
        'cache': {
            'x': x,
            'w_lm': w_lm
        }
    }

# Step 145 - full_model_forward
def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""
    caches = {}
    token_emb, token_cache = token_embedding_forward(
        x_ids, model_params["tok_emb"]
    )
    seq_len = x_ids.shape[1]
    pos_emb = slice_positional_embedding(
        model_params["pos_emb"], seq_len
    )
    x = add_token_and_positional_embeddings(token_emb, pos_emb)
    caches["emb"] = {
        "tok_cache": token_cache,
        "pos_emb": pos_emb,
        "seq_len": seq_len
    }
    x, block_caches = forward_through_all_blocks(
        x, model_params["blocks"]
    )
    caches["blocks"] = block_caches
    x, ln_cache = final_layernorm_forward(
        x,
        model_params["ln_f"]["gamma"],
        model_params["ln_f"]["beta"]
    )
    caches["ln_f"] = ln_cache
    lm_result = lm_head_linear_forward(
        x,
        model_params["lm_head"]["w_lm"],
        model_params["lm_head"]["b_lm"]
    )
    caches["lm_head"] = lm_result["cache"]
    return lm_result["logits"], caches

# Step 146 - full_model_backward
def full_model_backward(d_logits, caches, model_params):
    """Propagate d_logits back through LM head, final LN, blocks, and embeddings.

    Args:
        d_logits: (B, T, V) gradient w.r.t. the model output
        caches: nested dict from full_model_forward with keys
                'emb', 'blocks', 'ln_f', 'lm_head'
        model_params: nested dict matching the forward's parameter tree

    Returns:
        grads: nested dict mirroring model_params with keys
               'tok_emb', 'pos_emb', 'blocks', 'ln_f': {'gamma', 'beta'},
               'lm_head': {'w_lm', 'b_lm'}
    """
    lm_cache = caches["lm_head"]
    x_lm = lm_cache["x"]
    w_lm = lm_cache["w_lm"]
    B, T, D = x_lm.shape
    V = d_logits.shape[-1]
    x_2d = x_lm.reshape(B * T, D)
    d_logits_2d = d_logits.reshape(B * T, V)
    linear_cache = {"x": x_2d, "w": w_lm}
    d_x_2d = linear_backward_dx(d_logits_2d, linear_cache)
    d_w_lm = linear_backward_dw(d_logits_2d, linear_cache)
    d_b_lm = bias_add_backward_db(
        d_logits_2d,
        {"b_shape": model_params["lm_head"]["b_lm"].shape}
    )
    d_x = d_x_2d.reshape(B, T, D)
    ln_cache = caches["ln_f"]
    ln_cache_2d = {
        "x": ln_cache["x"].reshape(B * T, D),
        "x_hat": ln_cache["x_hat"].reshape(B * T, D),
        "mean": ln_cache["mean"].reshape(B * T, 1),
        "var": ln_cache["var"].reshape(B * T, 1),
        "gamma": ln_cache["gamma"],
        "eps": 1e-5,
    }
    ln_grads = layernorm_backward_implementation(
        d_x.reshape(B * T, D),
        ln_cache_2d
    )
    d_x = ln_grads["dx"].reshape(B, T, D)
    d_x, block_grads = backward_through_all_blocks(
        d_x,
        caches["blocks"],
        model_params["blocks"]
    )
    emb_grads = embedding_sum_backward(d_x)
    d_token = emb_grads["d_token_emb"]
    d_pos_used = emb_grads["d_pos_emb"]
    token_ids = caches["emb"]["tok_cache"]["token_ids"]
    d_tok_emb = np.zeros_like(model_params["tok_emb"])
    np.add.at(d_tok_emb, token_ids, d_token)
    seq_len = caches["emb"]["seq_len"]
    d_pos_emb = np.zeros_like(model_params["pos_emb"])
    d_pos_emb[:seq_len] = d_pos_used
    return {
        "tok_emb": d_tok_emb,
        "pos_emb": d_pos_emb,
        "blocks": block_grads,
        "ln_f": {
            "gamma": ln_grads["dgamma"],
            "beta": ln_grads["dbeta"],
        },
        "lm_head": {
            "w_lm": d_w_lm,
            "b_lm": d_b_lm,
        },
    }

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""
    def zeros_like_tree(tree):
        if isinstance(tree, dict):
            return {
                key: zeros_like_tree(value)
                for key, value in tree.items()
            }
        if isinstance(tree, list):
            return [
                zeros_like_tree(value)
                for value in tree
            ]
        if isinstance(tree, np.ndarray):
            return np.zeros_like(tree)
        return tree
    m = zeros_like_tree(model_params)
    v = zeros_like_tree(model_params)
    return m, v

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    return t + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    mt = beta1 * m + (1 - beta1) * grad
    return mt

# Step 151 - adam_update_second_moment
def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    vt = beta2 * v_prev + (1 - beta2) * grad ** 2
    return vt

# Step 152 - adam_bias_correction
def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    m = m / (1 - beta1 ** t)
    v = v / (1 - beta2 ** t)
    return [m, v]

# Step 153 - adam_parameter_update
import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    p_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p_new

# Step 154 - wire_full_training_loop
def wire_full_training_loop(
    params, train_ids, val_ids, block_size, batch_size,
    n_steps, lr, betas, eps
):
    """Run the full GPT training loop for n_steps and return (updated_params, history)."""
    beta1, beta2 = betas
    m, v = initialize_adam_moments(params)
    t = initialize_adam_step_counter()
    rng = np.random.default_rng()
    history = []
    def update_tree(p, g, mp, vp):
        if isinstance(p, dict):
            for key in p:
                update_tree(
                    p[key],
                    g[key],
                    mp[key],
                    vp[key],
                )
        elif isinstance(p, list):
            for i in range(len(p)):
                update_tree(
                    p[i],
                    g[i],
                    mp[i],
                    vp[i],
                )
        elif isinstance(p, np.ndarray):
            mp[...] = beta1 * mp + (1.0 - beta1) * g
            vp[...] = beta2 * vp + (1.0 - beta2) * (g ** 2)
            m_hat = mp / (1.0 - beta1 ** t)
            v_hat = vp / (1.0 - beta2 ** t)
            p[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            raise TypeError(
                f"Unsupported parameter type: {type(p)}"
            )
    for step in range(n_steps):
        x_ids, y_ids = get_batch(
            train_ids,
            block_size,
            batch_size,
            rng,
        )
        logits, caches = full_model_forward(
            x_ids,
            params,
        )
        B, T, V = logits.shape
        z = logits - np.max(
            logits,
            axis = -1,
            keepdims = True,
        )
        exp_z = np.exp(z)
        probs = exp_z / np.sum(
            exp_z,
            axis = -1,
            keepdims = True,
        )
        flat_probs = probs.reshape(-1, V)
        flat_y = y_ids.reshape(-1)
        N = flat_y.size
        loss = -np.mean(
            np.log(
                flat_probs[
                    np.arange(N),
                    flat_y,
                ] + 1e-12
            )
        )
        d_logits = flat_probs.copy()
        d_logits[
            np.arange(N),
            flat_y,
        ] -= 1.0
        d_logits /= N
        d_logits = d_logits.reshape(B, T, V)
        grads = full_model_backward(
            d_logits,
            caches,
            params,
        )
        t += 1
        update_tree(
            params,
            grads,
            m,
            v,
        )
        history.append({
            "step": step,
            "train_loss": float(loss),
        })
    return params, history

# Step 155 - logging_and_validation_loss
def logging_and_validation_loss(
    params, val_ids, block_size, batch_size, n_eval_batches
):
    """Estimate validation cross-entropy loss by averaging over several batches."""
    rng = np.random.default_rng()
    losses = []
    for _ in range(n_eval_batches):
        x_ids, y_ids = get_batch(
            val_ids,
            block_size,
            batch_size,
            rng,
        )
        logits, _ = full_model_forward(
            x_ids,
            params,
        )
        B, T, V = logits.shape
        z = logits - np.max(
            logits,
            axis = -1,
            keepdims = True,
        )
        exp_z = np.exp(z)
        probs = exp_z / np.sum(
            exp_z,
            axis = -1,
            keepdims = True,
        )
        flat_probs = probs.reshape(-1, V)
        flat_y = y_ids.reshape(-1)
        N = flat_y.size
        loss = -np.mean(
            np.log(
                flat_probs[
                    np.arange(N),
                    flat_y,
                ] + 1e-12
            )
        )
        losses.append(loss)
    return float(np.mean(losses))

# Step 156 - encode_prompt
import numpy as np

def encode_prompt(prompt, stoi):
    """Encode a string prompt to an int ndarray of shape (1, T)."""
    ids = encode_string(prompt, stoi)
    return np.array(ids, dtype = np.int64)[None, :]

# Step 157 - crop_context_to_block_size
def crop_context_to_block_size(context_ids, block_size):
    return context_ids[:, -block_size:]

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

