import time
import jax
import jax.numpy as jnp

try:
    import anthropic
except ImportError:
    anthropic = None
 
try:
    import openai
except ImportError:
    openai = None

from util import state_to_fen, build_user_prompt, get_legal_uci_moves, SYSTEM_PROMPT, ACTION_TO_UCI, parse_uci_from_response, uci_to_action_id

class AnthropicPlayer:

    def __init__(self, model: str = "claude-haiku-4-5", max_retries: int = 3):
        if anthropic is None:
            raise ImportError("pip install anthropic")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.name = f"Anthropic ({model})"
 
    def choose_move(self, state, move_history: list[str]) -> int:
        fen = state_to_fen(state)
        legal_moves = get_legal_uci_moves(state)
        color = "White" if int(state.current_player) == 0 else "Black"
        user_msg = build_user_prompt(fen, legal_moves, color, move_history)
 
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=32,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = resp.content[0].text
                uci = parse_uci_from_response(raw)
                if uci:
                    aid = uci_to_action_id(uci, state)
                    if aid is not None:
                        print(f"  [Anthropic] move: {uci}")
                        return aid
                    else:
                        print(f"  [Anthropic] illegal move '{uci}', retrying ({attempt+1}/{self.max_retries})")
                else:
                    print(f"  [Anthropic] could not parse '{raw}', retrying ({attempt+1}/{self.max_retries})")
            except Exception as e:
                print(f"  [Anthropic] API error: {e}, retrying ({attempt+1}/{self.max_retries})")
                time.sleep(2)
 
        # Fallback: pick a random legal move
        print("  [Anthropic] all retries failed, falling back to random move")
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        return int(jax.random.categorical(jax.random.PRNGKey(0), logits))

class OpenAIPlayer:
    def __init__(self, model: str = "gpt-5.4-nano", max_retries: int = 3):
        if openai is None:
            raise ImportError("pip install openai")
        self.client = openai.OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.name = f"OpenAI ({model})"
 
    def choose_move(self, state, move_history: list[str]) -> int:
        fen = state_to_fen(state)
        legal_moves = get_legal_uci_moves(state)
        color = "White" if int(state.current_player) == 0 else "Black"
        user_msg = build_user_prompt(fen, legal_moves, color, move_history)
 
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=32,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content
                uci = parse_uci_from_response(raw)
                if uci:
                    aid = uci_to_action_id(uci, state)
                    if aid is not None:
                        print(f"  [OpenAI] move: {uci}")
                        return aid
                    else:
                        print(f"  [OpenAI] illegal move '{uci}', retrying ({attempt+1}/{self.max_retries})")
                else:
                    print(f"  [OpenAI] could not parse '{raw}', retrying ({attempt+1}/{self.max_retries})")
            except Exception as e:
                print(f"  [OpenAI] API error: {e}, retrying ({attempt+1}/{self.max_retries})")
                time.sleep(2)
 
        print("  [OpenAI] all retries failed, falling back to random move")
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        return int(jax.random.categorical(jax.random.PRNGKey(1), logits))

 
class RandomPlayer:
    """Fallback random player for testing without API keys."""
    def __init__(self, seed: int = 42):
        self.rng = jax.random.PRNGKey(seed)
        self.name = "Random"
 
    def choose_move(self, state, move_history: list[str]) -> int:
        self.rng, sub = jax.random.split(self.rng)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
        action = int(jax.random.categorical(sub, logits))
        uci = ACTION_TO_UCI.get(action, "???")
        print(f"  [Random] move: {uci}")
        return action