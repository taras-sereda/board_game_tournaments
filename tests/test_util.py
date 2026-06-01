import jax
import pgx

from util import (
    UCI_TO_ACTION,
    _flip_uci,
    action_to_uci,
    get_legal_uci_moves,
    uci_to_action_id,
)


def _make_env():
    env = pgx.make("chess")
    init = jax.jit(env.init)
    step = jax.jit(env.step)
    return init, step


def test_uci_to_action_roundtrip_white_to_move():
    init, _ = _make_env()
    # current_player
    # observation (3d 8x8x119)
    # rewards (1d 2-element array)
    # board
    # board_history
    state = init(jax.random.key(0))

    action_id = uci_to_action_id("e2e4", state)
    assert action_id is not None
    assert bool(state.legal_action_mask[action_id])
    assert action_to_uci(action_id, state) == "e2e4"


def test_uci_to_action_roundtrip_black_to_move_after_white_move():
    init, step = _make_env()
    state = init(jax.random.key(0))

    white_action = uci_to_action_id("e2e4", state)
    assert white_action is not None
    state = step(state, white_action)
    breakpoint()
    black_action = uci_to_action_id("e7e5", state)
    assert black_action is not None
    assert bool(state.legal_action_mask[black_action])
    assert action_to_uci(black_action, state) == "e7e5"


def test_black_rotation_is_required_for_uci_to_pgx_lookup():
    init, step = _make_env()
    state = init(jax.random.key(0))

    white_action = uci_to_action_id("e2e4", state)
    assert white_action is not None
    state = step(state, white_action)

    absolute_key_action = UCI_TO_ACTION["e7e5"]
    assert not bool(state.legal_action_mask[absolute_key_action])

    mover_frame_key = _flip_uci("e7e5")
    mover_frame_action = UCI_TO_ACTION[mover_frame_key]
    assert bool(state.legal_action_mask[mover_frame_action])

    converted = uci_to_action_id("e7e5", state)
    assert converted == mover_frame_action


def test_legal_moves_are_real_board_coords_for_black_to_move():
    init, step = _make_env()
    state = init(jax.random.key(0))

    white_action = uci_to_action_id("e2e4", state)
    assert white_action is not None
    state = step(state, white_action)

    legal = get_legal_uci_moves(state)
    assert "e7e5" in legal
