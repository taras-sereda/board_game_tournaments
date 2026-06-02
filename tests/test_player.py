from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import pgx
import pytest

from player import AnthropicPlayer, _anthropic_response_text
from util import SYSTEM_PROMPT, uci_to_action_id


def _make_env():
    env = pgx.make("chess")
    init = jax.jit(env.init)
    return init


def _fake_anthropic_message(text: str, *, with_thinking: bool = False):
    blocks = []
    if with_thinking:
        blocks.append(SimpleNamespace(type="thinking", thinking="evaluating e2e4"))
    blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=blocks)


def _mock_stream_client(final_message):
    stream = MagicMock()
    stream.get_final_message.return_value = final_message
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)

    client = MagicMock()
    client.messages.stream.return_value = stream
    return client, stream


@pytest.fixture
def anthropic_player():
    with patch("player.anthropic") as mock_anthropic_module:
        mock_anthropic_module.Anthropic.return_value = MagicMock()
        player = AnthropicPlayer(model="claude-opus-4-8", max_tokens=4096)
        yield player


def test_anthropic_create_message_uses_streaming_api(anthropic_player):
    message = _fake_anthropic_message("e2e4")
    client, stream = _mock_stream_client(message)
    anthropic_player.client = client

    result = anthropic_player._create_message("pick a move")

    client.messages.stream.assert_called_once()
    client.messages.create.assert_not_called()
    stream.get_final_message.assert_called_once()
    assert result is message


def test_anthropic_stream_kwargs_include_adaptive_thinking():
    with patch("player.anthropic") as mock_anthropic_module:
        mock_anthropic_module.Anthropic.return_value = MagicMock()
        player = AnthropicPlayer(
            model="claude-opus-4-8",
            adaptive_thinking=True,
            effort="high",
            max_tokens=4096,
        )

    kwargs = player._message_kwargs("fen and legal moves")

    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["messages"] == [{"role": "user", "content": "fen and legal moves"}]
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


def test_anthropic_response_text_skips_thinking_block():
    message = _fake_anthropic_message("e7e5", with_thinking=True)
    assert _anthropic_response_text(message) == "e7e5"


def test_anthropic_choose_move_parses_streamed_text(anthropic_player):
    init = _make_env()
    state = init(jax.random.key(0))
    expected_action = uci_to_action_id("e2e4", state)

    message = _fake_anthropic_message("e2e4")
    client, _ = _mock_stream_client(message)
    anthropic_player.client = client

    action = anthropic_player.choose_move(state, [])

    assert action == expected_action
    client.messages.stream.assert_called_once()
