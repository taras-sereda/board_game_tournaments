from dataclasses import dataclass
import time
import jax
import jax.numpy as jnp
from pathlib import Path
from abc import abstractmethod
import json

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

from util import (
    UNC_ACTION,
    state_to_fen,
    build_user_prompt,
    get_legal_uci_moves,
    SYSTEM_PROMPT,
    ACTION_TO_UCI,
    parse_uci_from_response,
    uci_to_action_id,
)

ANTHROPIC_MODEL_OPUS_4_8 = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 64
DEFAULT_MAX_TOKENS_WITH_THINKING = 4096


def _anthropic_supports_effort(model: str) -> bool:
    return "haiku" not in model.lower()


def _anthropic_response_text(resp) -> str:
    for block in resp.content:
        if block.type == "text":
            return block.text
    raise ValueError("Anthropic response has no text block")


class Player:
    def __init__(self):
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def choose_move(self, state, move_history: list[str]) -> int:
        pass

    def dump_data(self, data: dict, dest_path: str | Path) -> str:
        if not isinstance(data, dict):
            data = data.to_dict()
        with open(dest_path, "w") as f:
            json.dump(data, f)


class AnthropicPlayer(Player):
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_retries: int = 3,
        out_dir: str | Path | None = None,
        adaptive_thinking: bool = False,
        effort: str = "low",
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.model_provider = "Anthropic"
        if anthropic is None:
            raise ImportError("pip install anthropic")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.adaptive_thinking = adaptive_thinking
        self.effort = effort
        if max_tokens is None:
            max_tokens = (
                DEFAULT_MAX_TOKENS_WITH_THINKING
                if adaptive_thinking
                else DEFAULT_MAX_TOKENS
            )
        self.max_tokens = max_tokens
        self.out_dir = out_dir
        if out_dir:
            log_dir = out_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = log_dir
        else:
            self.log_dir = None

    def _create_message(self, user_msg: str):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        }
        if self.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            if _anthropic_supports_effort(self.model):
                kwargs["output_config"] = {"effort": self.effort}
        return self.client.messages.create(**kwargs)

    def choose_move(self, state, move_history: list[str]) -> int:
        fen = state_to_fen(state)
        legal_moves = get_legal_uci_moves(state)
        color = "White" if fen.split()[1] == "w" else "Black"
        user_msg = build_user_prompt(fen, legal_moves, color, move_history)

        for attempt in range(self.max_retries):
            try:
                msg = {"role": "user", "content": user_msg}
                resp = self._create_message(user_msg)
                if self.log_dir:
                    req_path = (
                        self.log_dir / f"{self.name}_request_{int(time.time_ns())}.json"
                    )
                    resp_path = (
                        self.log_dir
                        / f"{self.name}_response_{int(time.time_ns())}.json"
                    )
                    self.dump_data(msg, req_path)
                    self.dump_data(resp, resp_path)
                raw = _anthropic_response_text(resp)
                uci = parse_uci_from_response(raw)
                if uci:
                    aid = uci_to_action_id(uci, state)
                    if aid is not None:
                        print(f"  [{self.model_provider}] move: {uci}")
                        return aid
                    else:
                        print(
                            f"  [{self.model_provider}] illegal move '{uci}', retrying ({attempt + 1}/{self.max_retries})"
                        )
                else:
                    print(
                        f"  [{self.model_provider}] could not parse '{raw}', retrying ({attempt + 1}/{self.max_retries})"
                    )
            except Exception as e:
                print(
                    f"  [{self.model_provider}] API error: {e}, retrying ({attempt + 1}/{self.max_retries})"
                )
                time.sleep(2)

        # Fallback: pick a random legal move
        print(
            f"  [{self.model_provider}] all retries failed, falling back to random move"
        )
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        key = jax.random.key(time.time_ns() % (2**32 - 1))
        return int(jax.random.categorical(key, logits))


class OpenAIPlayer(Player):
    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        endpoint: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        out_dir: str | Path | None = None,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
    ):
        super().__init__()
        self.model_provider = "GPT-OSS" if endpoint is not None else "OpenAI"
        if openai is None:
            raise ImportError("pip install openai")
        client_kwargs = {}
        if endpoint is not None:
            client_kwargs["base_url"] = endpoint
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        client_kwargs["_enforce_credentials"] = False
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        if max_completion_tokens is None:
            max_completion_tokens = (
                DEFAULT_MAX_TOKENS_WITH_THINKING
                if reasoning_effort is not None
                else DEFAULT_MAX_TOKENS
            )
        self.max_completion_tokens = max_completion_tokens
        self.out_dir = out_dir
        if out_dir:
            log_dir = out_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = log_dir
        else:
            self.log_dir = None

    def _create_completion(self, messages: list[dict]):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self.client.chat.completions.create(**kwargs)

    def choose_move(self, state, move_history: list[str]) -> int:
        fen = state_to_fen(state)
        legal_moves = get_legal_uci_moves(state)
        color = "White" if fen.split()[1] == "w" else "Black"
        user_msg = build_user_prompt(fen, legal_moves, color, move_history)

        for attempt in range(self.max_retries):
            try:
                msg = {"role": "user", "content": user_msg}
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    msg,
                ]
                resp = self._create_completion(messages)
                if self.log_dir:
                    req_path = (
                        self.log_dir / f"{self.name}_request_{int(time.time_ns())}.json"
                    )
                    resp_path = (
                        self.log_dir
                        / f"{self.name}_response_{int(time.time_ns())}.json"
                    )
                    self.dump_data(msg, req_path)
                    self.dump_data(resp, resp_path)
                raw = resp.choices[0].message.content
                uci = parse_uci_from_response(raw)
                if uci:
                    aid = uci_to_action_id(uci, state)
                    if aid is not None:
                        print(f"  [{self.model_provider}] move: {uci}")
                        return aid
                    else:
                        print(
                            f"  [{self.model_provider}] illegal move '{uci}', retrying ({attempt + 1}/{self.max_retries})"
                        )
                else:
                    print(
                        f"  [{self.model_provider}] could not parse '{raw}', retrying ({attempt + 1}/{self.max_retries})"
                    )
            except Exception as e:
                print(
                    f"  [{self.model_provider}] API error: {e}, retrying ({attempt + 1}/{self.max_retries})"
                )
                time.sleep(2)

        print(
            f"  [{self.model_provider}] all retries failed, falling back to random move"
        )
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        key = jax.random.key(time.time_ns() % (2**32 - 1))
        return int(jax.random.categorical(key, logits))


class RandomPlayer(Player):
    """Fallback random player for testing without API keys."""

    def __init__(self, seed: int = 42):
        self.rng = jax.random.key(seed)
        self.name = "Random"

    def choose_move(self, state, move_history: list[str]) -> int:
        self.rng, sub = jax.random.split(self.rng)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        action = int(jax.random.categorical(sub, logits))
        uci = ACTION_TO_UCI.get(action, UNC_ACTION)
        print(f"  [Random] move: {uci}")
        return action


@dataclass
class ModelProvider:
    provider: str
    model_name: str | None = None
    endpoint: str | None = None
    out_dir: str | Path | None = None
    anthropic_adaptive_thinking: bool = False
    anthropic_effort: str = "low"
    anthropic_max_tokens: int | None = None
    openai_reasoning_effort: str | None = None
    openai_max_completion_tokens: int | None = None


def make_player(model: ModelProvider, seed: int):
    if model.provider == "anthropic":
        return AnthropicPlayer(
            model=model.model_name,
            out_dir=model.out_dir,
            adaptive_thinking=model.anthropic_adaptive_thinking,
            effort=model.anthropic_effort,
            max_tokens=model.anthropic_max_tokens,
        )
    elif model.provider == "openai":
        return OpenAIPlayer(
            model=model.model_name,
            out_dir=model.out_dir,
            endpoint=model.endpoint,
            reasoning_effort=model.openai_reasoning_effort,
            max_completion_tokens=model.openai_max_completion_tokens,
        )
    elif model.provider == "gpt-oss":
        return OpenAIPlayer(
            model=model.model_name,
            out_dir=model.out_dir,
            endpoint=model.endpoint,
            reasoning_effort=model.openai_reasoning_effort,
            max_completion_tokens=model.openai_max_completion_tokens,
        )
    elif model.provider == "random":
        return RandomPlayer(seed=seed)
    else:
        raise ValueError(f"Unknown player type: {model.provider}")
