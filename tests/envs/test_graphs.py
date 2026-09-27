"""Tests for env_lib.utils.graphs."""

from __future__ import annotations

import numpy as np
import pytest

from env_lib.utils import graphs


@pytest.mark.parametrize("topology", graphs.TOPOLOGIES)
@pytest.mark.parametrize("n", [2, 7, 12])
def test_make_graph_is_symmetric_connected_and_loop_free(topology, n):
    if topology == "small_world" and n <= 4:
        pytest.skip("small-world graphs need more nodes than neighbours")
    adj = graphs.make_graph(topology, n, rng=np.random.default_rng(0), edge_probability=0.6)
    assert adj.shape == (n, n) and adj.dtype == bool
    assert np.array_equal(adj, adj.T)
    assert not adj.diagonal().any()
    assert graphs.is_connected(adj)


def test_make_graph_is_reproducible():
    a = graphs.make_graph("small_world", 20, rng=np.random.default_rng(3))
    b = graphs.make_graph("small_world", 20, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_known_structures():
    ring = graphs.make_graph("ring", 6)
    assert np.all(ring.sum(axis=1) == 2)
    star = graphs.make_graph("star", 5)
    assert star[0].sum() == 4 and star[1:, 1:].sum() == 0
    grid, pos = graphs.make_graph("grid", 12, return_positions=True)
    assert graphs.grid_shape(12) == (3, 4)
    assert grid.sum() // 2 == 3 * 3 + 2 * 4
    assert pos.shape == (12, 2)


def test_k_hop_adjacency_on_a_line():
    line = graphs.make_graph("line", 6)
    two = graphs.k_hop_adjacency(line, 2)
    assert two[0].tolist() == [False, True, True, False, False, False]
    three = graphs.k_hop_adjacency(line, 3, include_self=True)
    assert three[0].tolist() == [True, True, True, True, False, False]
    assert np.array_equal(graphs.k_hop_adjacency(line, 1), line)
    assert np.array_equal(graphs.k_hop_adjacency(line, 0, include_self=True), np.eye(6, dtype=bool))
    with pytest.raises(ValueError):
        graphs.k_hop_adjacency(line, -1)


def test_laplacian_and_connectivity():
    complete = graphs.make_graph("complete", 5)
    lap = graphs.laplacian(complete)
    assert np.allclose(lap.sum(axis=1), 0.0)
    assert graphs.algebraic_connectivity(complete) == pytest.approx(5.0)
    disconnected = np.zeros((4, 4), dtype=bool)
    disconnected[0, 1] = disconnected[1, 0] = True
    assert not graphs.is_connected(disconnected)
    assert graphs.algebraic_connectivity(disconnected) == pytest.approx(0.0, abs=1e-12)


def test_edges_neighbours_and_layouts():
    ring = graphs.make_graph("ring", 5)
    assert graphs.edge_list(ring).shape == (5, 2)
    assert [list(n) for n in graphs.neighbor_lists(ring)][0] == [1, 4]
    assert np.allclose(graphs.degree(ring), 2.0)
    pos = graphs.spring_layout(ring, iterations=50)
    assert pos.shape == (5, 2) and np.all(np.isfinite(pos)) and np.abs(pos).max() <= 1.0 + 1e-12
    assert np.allclose(np.linalg.norm(graphs.circular_layout(4), axis=1), 1.0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        graphs.make_graph("hypercube", 4)
    with pytest.raises(ValueError):
        graphs.make_graph("ring", 1)
    with pytest.raises(ValueError):
        graphs.make_graph("small_world", 10, neighbors=3)
    with pytest.raises(ValueError):
        graphs.make_graph("erdos_renyi", 30, edge_probability=0.001, max_attempts=3)
