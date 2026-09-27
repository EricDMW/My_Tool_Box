"""Tests for toolkit.neural_toolkit (networks, encoders, decoders, tabular tools, utilities)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import toolkit.neural_toolkit as nt
from toolkit.neural_toolkit.layers import (
    ACTIVATION_NAMES,
    PositionalEncoding,
    build_mlp,
    get_activation,
)

B, SEQ, OBS, ACT = 3, 5, 6, 4
TR = dict(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
RNN = dict(hidden_dim=8, num_layers=2)
CNN = dict(conv_dims=(4, 8), fc_dims=(16,))
ACTIVATION_TYPES = (
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Softplus,
)


def _image(batch: int = B, channels: int = 3, size: int = 12) -> torch.Tensor:
    return torch.randn(batch, channels, size, size)


def _output(result):
    """Strip the hidden state returned by recurrent modules."""
    return result[0] if isinstance(result, tuple) else result


# Each case: (factory, type, kwargs, input builder, expected output shape).
FACTORY_CASES = [
    (
        "policy",
        "mlp",
        dict(input_dim=OBS, output_dim=ACT, hidden_dims=(16, 16)),
        lambda: torch.randn(B, OBS),
        (B, ACT),
    ),
    ("policy", "cnn", dict(input_channels=3, output_dim=ACT, **CNN), _image, (B, ACT)),
    (
        "policy",
        "rnn",
        dict(input_dim=OBS, output_dim=ACT, **RNN),
        lambda: torch.randn(B, SEQ, OBS),
        (B, ACT),
    ),
    (
        "policy",
        "transformer",
        dict(input_dim=OBS, output_dim=ACT, **TR),
        lambda: torch.randn(B, SEQ, OBS),
        (B, ACT),
    ),
    ("value", "mlp", dict(input_dim=OBS, hidden_dims=(16,)), lambda: torch.randn(B, OBS), (B, 1)),
    ("value", "cnn", dict(input_channels=3, **CNN), _image, (B, 1)),
    (
        "value",
        "rnn",
        dict(input_dim=OBS, output_dim=2, **RNN),
        lambda: torch.randn(B, SEQ, OBS),
        (B, 2),
    ),
    ("value", "transformer", dict(input_dim=OBS, **TR), lambda: torch.randn(B, SEQ, OBS), (B, 1)),
    (
        "q",
        "mlp",
        dict(state_dim=OBS, action_dim=ACT, hidden_dims=(16,)),
        lambda: torch.randn(B, OBS),
        (B, ACT),
    ),
    (
        "q",
        "dueling",
        dict(state_dim=OBS, action_dim=ACT, hidden_dims=(16, 8)),
        lambda: torch.randn(B, OBS),
        (B, ACT),
    ),
    ("q", "cnn", dict(input_channels=3, action_dim=ACT, **CNN), _image, (B, ACT)),
    (
        "q",
        "rnn",
        dict(state_dim=OBS, action_dim=ACT, rnn_type="gru", **RNN),
        lambda: torch.randn(B, SEQ, OBS),
        (B, ACT),
    ),
    (
        "q",
        "transformer",
        dict(state_dim=OBS, action_dim=ACT, **TR),
        lambda: torch.randn(B, SEQ, OBS),
        (B, ACT),
    ),
    (
        "encoder",
        "mlp",
        dict(input_dim=OBS, latent_dim=5, hidden_dims=(16,)),
        lambda: torch.randn(B, OBS),
        (B, 5),
    ),
    ("encoder", "cnn", dict(input_channels=3, latent_dim=5, **CNN), _image, (B, 5)),
    (
        "encoder",
        "rnn",
        dict(input_dim=OBS, latent_dim=5, **RNN),
        lambda: torch.randn(B, SEQ, OBS),
        (B, 5),
    ),
    (
        "encoder",
        "transformer",
        dict(input_dim=OBS, latent_dim=5, **TR),
        lambda: torch.randn(B, SEQ, OBS),
        (B, 5),
    ),
    (
        "encoder",
        "vae",
        dict(input_dim=OBS, latent_dim=5, hidden_dims=(16,)),
        lambda: torch.randn(B, OBS),
        (B, 5),
    ),
    (
        "decoder",
        "mlp",
        dict(latent_dim=5, output_dim=OBS, hidden_dims=(16,)),
        lambda: torch.randn(B, 5),
        (B, OBS),
    ),
    (
        "decoder",
        "cnn",
        dict(latent_dim=5, output_channels=2, fc_dims=(16,), conv_dims=(8, 4), initial_size=3),
        lambda: torch.randn(B, 5),
        (B, 2, 12, 12),
    ),
    (
        "decoder",
        "rnn",
        dict(latent_dim=5, output_dim=OBS, max_seq_len=7, **RNN),
        lambda: torch.randn(B, 5),
        (B, 7, OBS),
    ),
    (
        "decoder",
        "transformer",
        dict(latent_dim=5, output_dim=OBS, max_seq_len=7, **TR),
        lambda: torch.randn(B, 5),
        (B, 7, OBS),
    ),
    (
        "decoder",
        "vae",
        dict(latent_dim=5, output_dim=OBS, hidden_dims=(16,)),
        lambda: torch.randn(B, 5),
        (B, OBS),
    ),
]

FACTORIES = {
    "policy": nt.PolicyFactory.create_policy,
    "value": nt.ValueFactory.create_value_network,
    "q": nt.QNetworkFactory.create_q_network,
    "encoder": nt.EncoderFactory.create_encoder,
    "decoder": nt.DecoderFactory.create_decoder,
}
CASE_IDS = [f"{family}-{kind}" for family, kind, *_ in FACTORY_CASES]


def _build(family: str, kind: str, kwargs: dict, **extra) -> nn.Module:
    return FACTORIES[family](kind, **kwargs, **extra)


# ---------------------------------------------------------------------------
# Exports and factories
# ---------------------------------------------------------------------------


def test_public_exports_and_aliases():
    for name in nt.__all__:
        assert hasattr(nt, name), name
    assert nt.CNNPolicyNetwork is nt.CNPolicyNetwork
    assert nt.RNNPolicyNetwork is nt.RNPolicyNetwork
    assert nt.CNNQNetwork is nt.CNQNetwork
    assert nt.RNNQNetwork is nt.RNQNetwork


@pytest.mark.parametrize("family,kind,kwargs,make_input,shape", FACTORY_CASES, ids=CASE_IDS)
def test_factory_forward_shapes(family, kind, kwargs, make_input, shape):
    net = _build(family, kind, kwargs)
    out = _output(net(make_input()))
    assert out.shape == shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("family", sorted(FACTORIES))
def test_factory_rejects_unknown_type(family):
    with pytest.raises(ValueError, match="valid types are: mlp"):
        FACTORIES[family]("does-not-exist")


def test_factory_type_is_case_insensitive():
    net = nt.PolicyFactory.create_policy("MLP", input_dim=3, output_dim=2)
    assert isinstance(net, nt.MLPPolicyNetwork)


def test_constructor_defaults_are_immutable():
    classes = [getattr(nt, name) for name in nt.__all__ if isinstance(getattr(nt, name), type)]
    for cls in classes:
        for param in inspect.signature(cls.__init__).parameters.values():
            assert not isinstance(param.default, (list, dict, set)), f"{cls.__name__}.{param.name}"


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ACTIVATION_NAMES)
def test_get_activation_returns_fresh_modules(name):
    a, b = get_activation(name), get_activation(name.upper())
    assert isinstance(a, nn.Module) and a is not b
    nn.Sequential(a)  # usable inside Sequential


def test_get_activation_aliases_and_errors():
    assert isinstance(get_activation("swish"), nn.SiLU)
    assert isinstance(get_activation("silu"), nn.SiLU)
    assert isinstance(get_activation("mish"), nn.Mish)
    assert isinstance(get_activation(nn.PReLU), nn.PReLU)
    with pytest.raises(ValueError, match="valid names are"):
        get_activation("relu6")
    with pytest.raises(TypeError):
        get_activation(3)
    with pytest.raises(ValueError, match="Unknown activation"):
        nt.NetworkUtils.get_activation("unknown")


def test_build_mlp_layer_order():
    mlp = build_mlp(4, [8, 8], output_dim=2, activation="tanh", dropout=0.1, layer_norm=True)
    kinds = [type(m) for m in mlp]
    block = [nn.Linear, nn.LayerNorm, nn.Tanh, nn.Dropout]
    assert kinds == block + block + [nn.Linear]
    hidden_only = build_mlp(4, [8])
    assert [type(m) for m in hidden_only] == [nn.Linear, nn.ReLU]
    assert len(build_mlp(4, [])) == 0


def _is_affine(fn, dim: int) -> bool:
    gen = torch.Generator().manual_seed(0)
    a, b = torch.randn(64, dim, generator=gen), torch.randn(64, dim, generator=gen)
    zero = torch.zeros(64, dim)
    with torch.no_grad():
        residual = fn(a + b) - fn(a) - fn(b) + fn(zero)
    return bool(torch.allclose(residual, torch.zeros_like(residual), atol=1e-5))


def test_deep_mlp_is_nonlinear():
    net = nt.MLPPolicyNetwork(OBS, ACT, hidden_dims=(32, 32), activation="relu")
    n_act = sum(isinstance(m, nn.ReLU) for m in net.feature_layers)
    assert n_act == 2
    assert not _is_affine(net, OBS)
    # Sanity check of the probe itself: a stack of linear layers is affine.
    linear_stack = nn.Sequential(nn.Linear(OBS, 32), nn.Linear(32, ACT))
    assert _is_affine(linear_stack, OBS)


@pytest.mark.parametrize("family,kind,kwargs,make_input,shape", FACTORY_CASES, ids=CASE_IDS)
def test_every_hidden_linear_is_followed_by_activation(family, kind, kwargs, make_input, shape):
    net = _build(family, kind, kwargs)
    for attr in ("feature_layers", "fc_layers", "state_layers", "shared_layers"):
        stack = getattr(net, attr, None)
        if stack is None:
            continue
        linears = sum(isinstance(m, nn.Linear) for m in stack)
        activations = sum(isinstance(m, ACTIVATION_TYPES) for m in stack)
        assert linears == activations == len(stack) // (2 + net.layer_norm + (net.dropout > 0))


def test_positional_encoding_varies_along_sequence_not_batch():
    pe = PositionalEncoding(8, dropout=0.0, max_len=16)
    out = pe(torch.zeros(4, 10, 8))
    assert torch.allclose(out[0], out[3])
    assert not torch.allclose(out[:, 0], out[:, 1])
    expected0 = torch.tensor([0.0, 1.0] * 4)  # sin(0), cos(0)
    assert torch.allclose(out[0, 0], expected0)
    assert "pe" not in pe.state_dict()
    with pytest.raises(ValueError, match="max_len=16"):
        pe(torch.zeros(1, 17, 8))
    with pytest.raises(ValueError, match="expects input of shape"):
        pe(torch.zeros(10, 8))


def test_positional_encoding_odd_dimension():
    pe = PositionalEncoding(7, dropout=0.0, max_len=4)
    assert pe(torch.zeros(2, 4, 7)).shape == (2, 4, 7)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (nt.TransformerPolicyNetwork, dict(input_dim=OBS, output_dim=ACT)),
        (nt.TransformerValueNetwork, dict(input_dim=OBS)),
        (nt.TransformerQNetwork, dict(state_dim=OBS, action_dim=ACT)),
        (nt.TransformerEncoder, dict(input_dim=OBS, latent_dim=5)),
    ],
)
def test_transformer_is_order_aware_and_batch_independent(cls, kwargs):
    net = cls(**kwargs, **TR).eval()
    x = torch.randn(4, SEQ, OBS)
    with torch.no_grad():
        batched = net(x)
        single = net(x[2:3])
        reversed_seq = net(x.flip(1))
    # A sample's output does not depend on its position in the batch.
    assert torch.allclose(batched[2:3], single, atol=1e-5)
    # Positional information is used: reversing the sequence changes the output.
    assert not torch.allclose(batched, reversed_seq, atol=1e-4)


def test_transformer_padding_mask_ignores_padded_positions():
    net = nt.TransformerPolicyNetwork(OBS, ACT, **TR).eval()
    x = torch.randn(2, SEQ, OBS)
    mask = torch.zeros(2, SEQ, dtype=torch.bool)
    mask[:, 3:] = True
    garbage = x.clone()
    garbage[:, 3:] = 100.0
    with torch.no_grad():
        a, b = net(x, mask=mask), net(garbage, mask=mask)
        truncated = net(x[:, :3])
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, truncated, atol=1e-5)
    with pytest.raises(ValueError, match="mask must have shape"):
        net(x, mask=mask[:, :2])


def test_recurrent_hidden_state_can_be_carried():
    net = nt.RNPolicyNetwork(OBS, ACT, **RNN).eval()
    x = torch.randn(B, SEQ, OBS)
    with torch.no_grad():
        full, _ = net(x)
        hidden = None
        for t in range(SEQ):
            step, hidden = net(x[:, t], hidden)
    assert torch.allclose(full, step, atol=1e-5)
    assert hidden[0].shape == (RNN["num_layers"], B, RNN["hidden_dim"])


def test_bidirectional_rnn_uses_both_final_states():
    net = nt.RNNEncoder(OBS, 5, hidden_dim=8, num_layers=1, bidirectional=True, fc_dims=())
    assert net.latent_layer.in_features == 16
    x = torch.randn(B, SEQ, OBS)
    with torch.no_grad():
        latent, (h_n, _) = net(x)
        expected = net.latent_layer(torch.cat([h_n[0], h_n[1]], dim=-1))
    assert torch.allclose(latent, expected)


# ---------------------------------------------------------------------------
# Q-networks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mlp", "dueling", "cnn", "rnn", "transformer"])
def test_q_network_action_selection(kind):
    family, _, kwargs, make_input, _ = next(c for c in FACTORY_CASES if c[:2] == ("q", kind))
    net = _build(family, kind, kwargs).eval()
    state = make_input()
    action = torch.tensor([0, 3, 1])
    with torch.no_grad():
        q_all = _output(net(state))
        q_sa = _output(net(state, action))
        q_sa_col = _output(net(state, action.unsqueeze(-1)))
    assert q_sa.shape == (B,)
    assert torch.allclose(q_sa, q_all[torch.arange(B), action])
    assert torch.allclose(q_sa, q_sa_col)
    with pytest.raises(ValueError, match="one index per state"):
        net(state, torch.tensor([0, 1]))


def test_dueling_aggregation():
    net = nt.DuelingQNetwork(OBS, ACT, hidden_dims=(16, 8))
    state = torch.randn(B, OBS)
    with torch.no_grad():
        q = net(state)
        value = net.value_layers(net.shared_layers(state)).squeeze(-1)
    # With the mean-subtracted advantage, the mean Q-value equals V(s).
    assert torch.allclose(q.mean(dim=-1), value, atol=1e-6)
    single_hidden = nt.DuelingQNetwork(OBS, ACT, hidden_dims=(8,))
    assert len(single_hidden.shared_layers) == 0
    assert single_hidden(state).shape == (B, ACT)
    with pytest.raises(ValueError, match="at least one entry"):
        nt.DuelingQNetwork(OBS, ACT, hidden_dims=())


def test_continuous_action_critic():
    critic = nt.MLPQNetwork(OBS, 2, hidden_dims=(16,), continuous_action=True)
    assert critic.state_layers[0].in_features == OBS + 2
    out = critic(torch.randn(B, OBS), torch.randn(B, 2))
    assert out.shape == (B, 1)
    with pytest.raises(ValueError, match="requires an action"):
        critic(torch.randn(B, OBS))
    with pytest.raises(ValueError, match="action must have shape"):
        critic(torch.randn(B, OBS), torch.randn(B, 3))


# ---------------------------------------------------------------------------
# Convolutional networks, encoders and decoders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["tanh", "elu", "gelu", "swish"])
def test_cnn_accepts_any_activation(activation):
    net = nt.CNPolicyNetwork(3, ACT, activation=activation, **CNN)
    assert net(_image()).shape == (B, ACT)


def test_cnn_encoder_default_architecture_on_64x64_and_other_sizes():
    enc = nt.CNNEncoder(3, 8)
    assert enc.strides == [1, 2, 2, 2]
    assert enc(torch.randn(2, 3, 64, 64)).shape == (2, 8)
    assert enc(torch.randn(2, 3, 40, 40)).shape == (2, 8)


def test_cnn_per_layer_argument_validation():
    with pytest.raises(ValueError, match="kernel_sizes must have one entry"):
        nt.CNPolicyNetwork(3, ACT, conv_dims=(4, 8), kernel_sizes=(3, 3, 3))
    net = nt.CNPolicyNetwork(3, ACT, conv_dims=(4, 8), kernel_sizes=5, strides=2)
    assert net.kernel_sizes == [5, 5] and net.strides == [2, 2]
    with pytest.raises(ValueError, match="images of shape"):
        net(torch.randn(3, 12, 12))


@pytest.mark.parametrize(
    "conv_dims,kernel_sizes,strides,initial_size,expected",
    [
        ((16, 8, 8, 4), None, None, 4, 64),
        ((8, 4), 4, 2, 3, 12),
        ((8, 6, 4), (3, 5, 2), (1, 2, 3), 5, 30),
    ],
)
def test_cnn_decoder_output_size(conv_dims, kernel_sizes, strides, initial_size, expected):
    dec = nt.CNNDecoder(
        5,
        3,
        fc_dims=(16,),
        conv_dims=conv_dims,
        kernel_sizes=kernel_sizes,
        strides=strides,
        initial_size=initial_size,
    )
    out = dec(torch.randn(2, 5))
    assert dec.output_size == expected
    assert out.shape == (2, 3, expected, expected)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output
    n_deconv = sum(isinstance(m, nn.ConvTranspose2d) for m in dec.conv_layers)
    assert n_deconv == len(conv_dims)


def test_cnn_decoder_options():
    linear = nt.CNNDecoder(5, 3, fc_dims=(), conv_dims=(4,), initial_size=2, output_activation=None)
    assert isinstance(linear.conv_layers[-1], nn.ConvTranspose2d)
    assert isinstance(linear.initial_activation, nn.ReLU)
    with pytest.raises(ValueError, match="cannot upsample"):
        nt.CNNDecoder(5, 3, conv_dims=(4,), kernel_sizes=4, strides=1)


def test_sequence_decoders_respect_seq_len():
    rnn = nt.RNNDecoder(5, OBS, max_seq_len=4, **RNN)
    tr = nt.TransformerDecoder(5, OBS, max_seq_len=4, **TR)
    z = torch.randn(2, 5)
    for dec in (rnn, tr):
        assert dec(z).shape == (2, 4, OBS)
        assert dec(z, seq_len=9).shape == (2, 9, OBS)
        with pytest.raises(ValueError):
            dec(z, seq_len=0)


def test_variational_encoder_reparameterisation():
    enc = nt.VariationalEncoder(OBS, 4, hidden_dims=(16,))
    x = torch.randn(B, OBS)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        z, mu, logvar = enc(x)
        torch.manual_seed(0)
        eps = torch.randn_like(mu)
    assert torch.allclose(z, mu + eps * torch.exp(0.5 * logvar))
    kl = nt.VariationalEncoder.kl_divergence(torch.zeros(2, 4), torch.zeros(2, 4))
    assert torch.allclose(kl, torch.zeros(2))
    (z.sum() + nt.VariationalEncoder.kl_divergence(mu, logvar).sum()).backward()
    assert enc.mu_layer.weight.grad is not None and enc.logvar_layer.weight.grad is not None


# ---------------------------------------------------------------------------
# Device, gradients and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,kind,kwargs,make_input,shape", FACTORY_CASES, ids=CASE_IDS)
def test_device_argument_moves_parameters_and_buffers(family, kind, kwargs, make_input, shape):
    net = _build(family, kind, kwargs, device="meta")
    assert net.device == "meta"
    tensors = list(net.parameters()) + list(net.buffers())
    assert tensors and all(t.device.type == "meta" for t in tensors)


@pytest.mark.parametrize("family,kind,kwargs,make_input,shape", FACTORY_CASES, ids=CASE_IDS)
def test_gradients_reach_every_parameter(family, kind, kwargs, make_input, shape):
    net = _build(family, kind, kwargs)
    result = net(make_input())
    out = result[0] if isinstance(result, tuple) else result
    out.pow(2).mean().backward()
    missing = [name for name, p in net.named_parameters() if p.grad is None]
    assert not missing, missing


def test_constructor_validation():
    with pytest.raises(ValueError, match="divisible by nhead"):
        nt.TransformerPolicyNetwork(OBS, ACT, d_model=10, nhead=3)
    with pytest.raises(ValueError, match="dropout"):
        nt.MLPPolicyNetwork(OBS, ACT, dropout=1.5)
    with pytest.raises(ValueError, match="rnn_type"):
        nt.RNPolicyNetwork(OBS, ACT, rnn_type="transformer")
    with pytest.raises(TypeError, match="hidden_dims"):
        nt.MLPValueNetwork(OBS, hidden_dims=64)
    with pytest.raises(ValueError, match="output_dim"):
        nt.MLPPolicyNetwork(OBS, 0)
    with pytest.raises(ValueError, match="max_len"):
        nt.TransformerValueNetwork(OBS, max_len=4, **TR)(torch.randn(1, 5, OBS))


def test_legacy_get_activation_method_returns_module():
    net = nt.MLPPolicyNetwork(OBS, ACT, activation="elu")
    act = net._get_activation()
    assert isinstance(act, nn.ELU)
    assert act(torch.tensor([-1.0])).item() < 0


# ---------------------------------------------------------------------------
# Tabular tools
# ---------------------------------------------------------------------------


def test_tables_do_not_touch_global_rng():
    state = np.random.get_state()
    q = nt.QTable(4, 3, rng=0)
    p = nt.PolicyTable(4, 3, rng=0)
    for _ in range(20):
        q.get_policy(0, epsilon=0.5)
        q.get_softmax_policy(0, temperature=1.0)
        p.get_policy(1)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_seeded_rng_is_reproducible():
    def actions(seed):
        q = nt.DiscreteTools.create_q_table(3, 5, rng=seed)
        return [q.get_policy(0, epsilon=0.7) for _ in range(30)]

    assert actions(1) == actions(1)
    assert actions(1) != actions(2)
    gen = np.random.default_rng(3)
    assert nt.QTable(2, 2, rng=gen).rng is gen


def test_epsilon_greedy_extremes_and_probs():
    q = nt.QTable(2, 4, rng=0)
    q.set_value(0, 1.0, 2)
    assert all(q.get_policy(0, epsilon=0.0) == 2 for _ in range(10))
    counts = np.bincount([q.get_policy(0, epsilon=1.0) for _ in range(2000)], minlength=4)
    assert counts.min() > 400
    probs = q.get_epsilon_greedy_probs(0, epsilon=0.2)
    assert np.allclose(probs, [0.05, 0.05, 0.85, 0.05])
    with pytest.raises(ValueError, match="epsilon"):
        q.get_policy(0, epsilon=1.5)


def test_softmax_policy_is_numerically_stable():
    q = nt.QTable(1, 3, rng=0)
    q.table[0] = [1000.0, 1001.0, 999.0]
    probs = q.get_softmax_probs(0, temperature=1.0)
    ref = np.exp([-1.0, 0.0, -2.0])
    assert np.allclose(probs, ref / ref.sum())
    assert q.get_softmax_policy(0, temperature=1e-3) == 1
    assert nt.DiscreteTools.boltzmann_policy(q, 0, 1e-3) == 1
    with pytest.raises(ValueError, match="temperature"):
        q.get_softmax_policy(0, temperature=0.0)


def test_td_updates():
    q = nt.QTable(3, 2)
    q.set_value(1, 2.0, 0)
    q.set_value(1, 4.0, 1)
    nt.DiscreteTools.q_learning_update(q, 0, 0, reward=1.0, next_state=1, gamma=0.5, alpha=0.5)
    assert q.get_value(0, 0) == pytest.approx(0.5 * (1.0 + 0.5 * 4.0))
    nt.DiscreteTools.q_learning_update(
        q, 2, 0, reward=1.0, next_state=1, gamma=0.5, alpha=1.0, done=True
    )
    assert q.get_value(2, 0) == pytest.approx(1.0)
    nt.DiscreteTools.sarsa_update(
        q, 2, 1, reward=0.0, next_state=1, next_action=0, gamma=0.5, alpha=1.0
    )
    assert q.get_value(2, 1) == pytest.approx(1.0)


def test_expected_sarsa_matches_reference():
    q = nt.QTable(2, 3)
    q.table[1] = [1.0, 3.0, 2.0]
    policy = nt.PolicyTable(2, 3)
    policy.set_policy_probs(1, [0.2, 0.5, 0.3])
    nt.DiscreteTools.expected_sarsa_update(q, 0, 0, 1.0, 1, policy, gamma=0.9, alpha=1.0)
    assert q.get_value(0, 0) == pytest.approx(1.0 + 0.9 * (0.2 * 1 + 0.5 * 3 + 0.3 * 2))
    eps = 0.3
    nt.DiscreteTools.expected_sarsa_update(q, 0, 1, 0.0, 1, gamma=1.0, alpha=1.0, epsilon=eps)
    expected = (eps / 3) * (1 + 3 + 2) + (1 - eps) * 3.0
    assert q.get_value(0, 1) == pytest.approx(expected)
    with pytest.raises(ValueError, match="same number of actions"):
        nt.DiscreteTools.expected_sarsa_update(q, 0, 0, 0.0, 1, nt.PolicyTable(2, 4))


def test_ucb_policy():
    q = nt.QTable(2, 3, rng=0)
    counts = np.zeros((2, 3))
    counts[0] = [5, 0, 2]
    assert nt.DiscreteTools.ucb_policy(q, 0, counts) == 1  # untried action first
    counts[0] = [10, 1, 5]
    q.table[0] = [0.5, 0.0, 0.4]
    c = 2.0
    ucb = q.table[0] + c * np.sqrt(np.log(16) / counts[0])
    assert nt.DiscreteTools.ucb_policy(q, 0, counts, c) == int(np.argmax(ucb))
    with pytest.raises(ValueError, match="visit_counts"):
        nt.DiscreteTools.ucb_policy(q, 0, np.ones(3))


def test_thompson_sampling_policy():
    q = nt.QTable(1, 3, rng=0)
    q.table[0] = [-3.0, 3.0, 0.0]
    counts = np.full((1, 3), 1000.0)
    picks = [nt.DiscreteTools.thompson_sampling_policy(q, 0, counts) for _ in range(20)]
    assert set(picks) == {1}
    q.table[0] = [1e4, -1e4, 0.0]  # no overflow warnings with extreme values
    assert nt.DiscreteTools.thompson_sampling_policy(q, 0, counts) == 0


def test_policy_table_normalisation():
    p = nt.PolicyTable(2, 4, rng=0)
    assert np.allclose(p.get_policy_probs(0), 0.25)
    p.set_value(0, 0.7, 1)
    probs = p.get_policy_probs(0)
    assert probs[1] == pytest.approx(0.7)
    assert probs.sum() == pytest.approx(1.0)
    assert np.allclose(probs[[0, 2, 3]], 0.1)
    p.update_value(0, 1.0, 1, learning_rate=0.5)
    assert p.get_value(0, 1) == pytest.approx(0.85)
    assert p.get_policy_probs(0).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        p.set_value(0, 1.5, 1)
    p.set_deterministic_policy(1, 3)
    assert all(p.get_policy(1) == 3 for _ in range(10))
    p.set_value(1, 0.5, 0)  # the other actions are rescaled proportionally
    assert np.allclose(p.get_policy_probs(1), [0.5, 0.0, 0.0, 0.5])
    freq = np.bincount([p.get_policy(1) for _ in range(4000)], minlength=4) / 4000
    assert np.allclose(freq, [0.5, 0, 0, 0.5], atol=0.05)
    p.set_deterministic_policy(1, 0)
    p.set_value(1, 0.4, 0)  # other actions had zero mass: the rest is shared uniformly
    assert np.allclose(p.get_policy_probs(1), [0.4, 0.2, 0.2, 0.2])
    p.table[1] = [2.0, 0.0, 1.0, 1.0]
    p._normalize_state(1)
    assert np.allclose(p.get_policy_probs(1), [0.5, 0.0, 0.25, 0.25])


def test_value_table_and_iteration_helpers():
    v = nt.DiscreteTools.create_value_table(3, initial_value=1.0)
    v.update_value(0, 3.0, learning_rate=0.5)
    assert v.get_value(0) == pytest.approx(2.0)
    q = nt.QTable(3, 2)
    q.table[2] = [0.1, 0.9]
    nt.DiscreteTools.value_iteration_update(v, 2, q)
    assert v.get_values()[2] == pytest.approx(0.9)
    pol = nt.DiscreteTools.create_policy_table(3, 2)
    nt.DiscreteTools.policy_iteration_update(pol, 2, q)
    assert np.allclose(pol.get_policy_probs(2), [0.0, 1.0])


def test_index_and_size_validation():
    q = nt.QTable(3, 2)
    with pytest.raises(IndexError):
        q.get_value(-1, 0)
    with pytest.raises(IndexError):
        q.get_value(0, 2)
    with pytest.raises(TypeError):
        q.get_value(0.5, 0)
    with pytest.raises(ValueError, match="action_space_size"):
        nt.QTable(3)
    with pytest.raises(ValueError, match="state_space_size"):
        nt.ValueTable(0)
    with pytest.raises(TypeError):
        nt.BaseDiscreteTable(3)


def test_discrete_environment_subclass():
    class Chain(nt.DiscreteEnvironment):
        def reset(self):
            self.current_state = 0
            return 0

        def step(self, action):
            self.current_state = min(self.current_state + 1, self.state_space_size - 1)
            done = self.current_state == self.state_space_size - 1
            return self.current_state, float(done), done, {}

    env = Chain(3, 2, rng=0)
    assert env.reset() == 0
    assert env.step(0)[0] == 1 and env.get_state() == 1
    with pytest.raises(TypeError):
        nt.DiscreteEnvironment(3, 2)


# ---------------------------------------------------------------------------
# NetworkUtils
# ---------------------------------------------------------------------------


def _small_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.ReLU(), nn.Linear(8, 2))


def test_initialize_weights_is_recursive():
    model = _small_model()
    nt.NetworkUtils.initialize_weights(model, method="uniform")
    for layer in (model[0], model[3]):
        assert layer.weight.abs().max() <= 0.1
        assert torch.count_nonzero(layer.bias) == 0
    with pytest.raises(ValueError, match="valid methods"):
        nt.NetworkUtils.initialize_weights(model, method="zeros")


def test_parameter_counting():
    model = _small_model()
    by_layer = nt.NetworkUtils.count_parameters_by_layer(model)
    assert (
        sum(by_layer.values())
        == nt.NetworkUtils.count_parameters(model)
        == 4 * 8 + 8 + 16 + 8 * 2 + 2
    )
    nt.NetworkUtils.freeze_layers(model, ["0."])
    assert nt.NetworkUtils.count_parameters(model) == 16 + 18
    nt.NetworkUtils.unfreeze_layers(model, ["0."])
    assert nt.NetworkUtils.count_parameters(model) == 74


def test_weight_decay_groups_and_optimizers():
    model = _small_model()
    groups = nt.NetworkUtils.apply_weight_decay(model, 0.1)
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert {id(p) for p in decay["params"]} == {id(model[0].weight), id(model[3].weight)}
    assert len(no_decay["params"]) == 4  # two biases + LayerNorm weight and bias
    only_bias = nn.Sequential(nn.LayerNorm(3))
    assert len(nt.NetworkUtils.apply_weight_decay(only_bias, 0.1)) == 1
    opt = nt.NetworkUtils.create_optimizer(model, "adamw", lr=1e-3, weight_decay=0.1)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    sgd = nt.NetworkUtils.create_optimizer(model, "SGD", lr=0.1)
    assert sgd.param_groups[0]["momentum"] == 0.9
    with pytest.raises(ValueError, match="valid options"):
        nt.NetworkUtils.create_optimizer(model, "lamb")
    sched = nt.NetworkUtils.create_scheduler(opt, "cosine", T_max=10)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    with pytest.raises(ValueError, match="valid options"):
        nt.NetworkUtils.create_scheduler(opt, "warmup")


def test_grad_norm_helpers():
    model = _small_model()
    assert nt.NetworkUtils.get_grad_norm(model) == 0.0
    model(torch.randn(5, 4)).sum().backward()
    expected = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert nt.NetworkUtils.get_grad_norm(model) == pytest.approx(float(expected), rel=1e-5)
    inf_norm = max(float(p.grad.abs().max()) for p in model.parameters())
    assert nt.NetworkUtils.get_grad_norm(model, float("inf")) == pytest.approx(inf_norm)
    total = nt.NetworkUtils.clip_grad_norm(model, max_norm=1e-3)
    assert total == pytest.approx(float(expected), rel=1e-5)
    assert nt.NetworkUtils.get_grad_norm(model) == pytest.approx(1e-3, rel=1e-3)


def test_checkpoint_round_trip(tmp_path):
    model = nt.MLPPolicyNetwork(4, 2, hidden_dims=(8,))
    opt = nt.NetworkUtils.create_optimizer(model, "adam")
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    path = nt.NetworkUtils.save_checkpoint(
        model, opt, 7, 0.25, tmp_path / "ckpt" / "model.pt", note="x"
    )
    assert path.exists()
    clone = nt.MLPPolicyNetwork(4, 2, hidden_dims=(8,))
    clone_opt = nt.NetworkUtils.create_optimizer(clone, "adam")
    epoch, loss = nt.NetworkUtils.load_checkpoint(clone, clone_opt, path)
    assert (epoch, loss) == (7, 0.25)
    for a, b in zip(model.parameters(), clone.parameters()):
        assert torch.equal(a, b)
    assert clone_opt.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
    epoch, _ = nt.NetworkUtils.load_checkpoint(clone, None, path, map_location="cpu")
    assert epoch == 7


def test_conv_shape_helpers_match_pytorch():
    x = torch.zeros(1, 1, 37, 23)
    conv = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=1, dilation=2)
    out = conv(x)
    assert nt.NetworkUtils.compute_output_size(37, 5, 2, 1, 2) == out.shape[2]
    assert nt.NetworkUtils.compute_output_size(23, 5, 2, 1, 2) == out.shape[3]
    layers = [dict(kernel_size=3, stride=2), dict(kernel_size=5, stride=1, padding=0)]
    stack = nn.Sequential(nn.Conv2d(1, 1, 3, 2, 1), nn.Conv2d(1, 1, 5, 1, 0))
    assert nt.NetworkUtils.compute_conv_output_size((37, 23), layers) == tuple(stack(x).shape[2:])
    deconv = nn.ConvTranspose2d(1, 1, 4, stride=3, padding=1, output_padding=2)
    assert (
        nt.NetworkUtils.compute_transposed_output_size(9, 4, 3, 1, 2)
        == deconv(torch.zeros(1, 1, 9, 9)).shape[2]
    )
    with pytest.raises(ValueError, match="empty output"):
        nt.NetworkUtils.compute_output_size(2, 5)


def test_network_utils_builders():
    mlp = nt.NetworkUtils.create_mlp_layers(4, [8, 8], 2, activation="gelu", layer_norm=True)
    assert sum(isinstance(m, nn.GELU) for m in mlp) == 2
    assert mlp(torch.randn(3, 4)).shape == (3, 2)
    conv = nt.NetworkUtils.create_conv_layers(3, [4, 8], activation="tanh", strides=[1, 2])
    assert conv(torch.randn(1, 3, 8, 8)).shape == (1, 8, 4, 4)
    conv_pad = nt.NetworkUtils.create_conv_layers(3, [4], padding=[0])
    assert conv_pad(torch.randn(1, 3, 8, 8)).shape == (1, 4, 6, 6)
    with pytest.raises(ValueError, match="kernel_sizes must have 2 entries"):
        nt.NetworkUtils.create_conv_layers(3, [4, 8], kernel_sizes=[3])
    deconv = nt.NetworkUtils.create_transposed_conv_layers(4, [4, 2], activation="elu")
    assert deconv(torch.randn(1, 4, 5, 5)).shape == (1, 2, 20, 20)
    pe = nt.NetworkUtils.create_positional_encoding(8, max_len=10)
    assert pe.shape == (1, 10, 8) and not pe.requires_grad
    assert torch.allclose(pe, PositionalEncoding(8, 0.0, max_len=10).pe)
    with pytest.raises(ValueError, match="divisible"):
        nt.NetworkUtils.create_attention_layer(10, num_heads=3)
