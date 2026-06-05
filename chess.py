import argparse
import json
import jax
import pgx

from pathlib import Path

from util import state_to_fen, action_to_uci
from player import ANTHROPIC_MODEL_OPUS_4_8, make_player, ModelProvider

env = pgx.make("chess")
init = jax.jit(env.init)
step = jax.jit(env.step)


def _anthropic_provider(args, out_dir: Path) -> ModelProvider:
    return ModelProvider(
        provider="anthropic",
        model_name=args.anthropic_model,
        out_dir=out_dir,
        anthropic_adaptive_thinking=args.anthropic_adaptive_thinking,
        anthropic_effort=args.anthropic_effort,
        anthropic_max_tokens=args.anthropic_max_tokens,
    )


def _openai_provider(args, out_dir: Path, provider: str) -> ModelProvider:
    return ModelProvider(
        provider=provider,
        model_name=args.openai_model,
        endpoint=args.openai_endpoint,
        out_dir=out_dir,
        openai_reasoning_effort=args.openai_reasoning_effort,
        openai_max_completion_tokens=args.openai_max_completion_tokens,
    )


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.white == "anthropic":
        white_model = _anthropic_provider(args, out_dir)
    elif args.white == "openai":
        white_model = _openai_provider(args, out_dir, "openai")
    elif args.white == "gpt-oss":
        white_model = _openai_provider(args, out_dir, "gpt-oss")
    elif args.white == "random":
        white_model = ModelProvider(provider=args.white, out_dir=out_dir)

    if args.black == "anthropic":
        black_model = _anthropic_provider(args, out_dir)
    elif args.black == "openai":
        black_model = _openai_provider(args, out_dir, "openai")
    elif args.black == "gpt-oss":
        black_model = _openai_provider(args, out_dir, "gpt-oss")
    elif args.black == "random":
        black_model = ModelProvider(provider=args.black, out_dir=out_dir)

    rng = jax.random.key(args.seed)
    rng, sub = jax.random.split(rng)
    white_player = make_player(white_model, jax.random.key_data(sub)[0].item())
    rng, sub = jax.random.split(rng)
    black_player = make_player(black_model, jax.random.key_data(sub)[0].item())
    # Set up players: index 0 = white, index 1 = black
    players = [white_player, black_player]

    print(f"White: {white_player.name}")
    print(f"Black: {black_player.name}")
    print()

    state = init(sub)
    white_agent = int(state.current_player)
    move_history: list[str] = []
    idx = 0
    max_moves = args.max_moves if args.max_moves is not None else int(1e3)

    while not state.terminated and idx < max_moves:
        fen = state_to_fen(state)
        side = fen.split()[1]  # "w" or "b" — authoritative
        color_name = "White" if side == "w" else "Black"
        player = white_player if side == "w" else black_player
        action = player.choose_move(state, move_history)
        print(f"Move {idx + 1} ({color_name} / {player.name}) — FEN: {fen}")
        uci = action_to_uci(action, state)  # state is still pre-step here
        move_history.append(uci)
        state = step(state, action)
        if not args.no_save:
            state.save_svg(f"{out_dir}/{args.save_prefix}{idx:04}.svg")
        idx += 1

    print()
    print(f"Game over after {idx} moves.")
    rewards = state.rewards
    if rewards[0] == -1 and rewards[1] == 1:
        print("Result: White wins!")
    elif rewards[0] == 1 and rewards[1] == -1:
        print("Result: Black wins!")
    else:
        print("Result: Draw")
    print(f"Rewards: {rewards}")
    print(f"Move history: {' '.join(move_history)}")

    if not args.no_save:
        state.save_svg(f"{out_dir}/{args.save_prefix}final.svg")
        # Save game log
        white_reward = float(state.rewards[white_agent])
        result = (
            "white" if white_reward == 1 else "black" if white_reward == -1 else "draw"
        )
        log = {
            "white": white_player.name,
            "black": black_player.name,
            "result": result,
            "moves": move_history,
            "final_fen": state_to_fen(state),
        }
        with open(f"{out_dir}/game_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nSaved {idx} SVGs + final + game_log.json to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM vs LLM chess via PGX")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument(
        "--save-prefix", type=str, default="state_", help="Prefix for saved SVGs"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./states", help="Directory to save SVG files"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save SVG files during play"
    )

    parser.add_argument(
        "--white",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "gpt-oss", "random"],
        help="Player for White (default: anthropic)",
    )
    parser.add_argument(
        "--black",
        type=str,
        default="openai",
        choices=["anthropic", "openai", "gpt-oss", "random"],
        help="Player for Black (default: openai)",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-haiku-4-5",
        help=(
            "Anthropic model id (e.g. claude-haiku-4-5, "
            f"{ANTHROPIC_MODEL_OPUS_4_8})"
        ),
    )
    parser.add_argument(
        "--anthropic-adaptive-thinking",
        action="store_true",
        help=(
            "Enable adaptive extended thinking (required style on Opus 4.7+ "
            "when thinking is on; increases max_tokens budget)"
        ),
    )
    parser.add_argument(
        "--anthropic-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Thinking depth for Opus/Sonnet when adaptive thinking is enabled",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=None,
        help="Output cap (default 64, or 4096 with --anthropic-adaptive-thinking)",
    )
    parser.add_argument(
        "--openai-model", type=str, default="gpt-5.4-nano", help="OpenAI model name"
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning models (omit to disable)",
    )
    parser.add_argument(
        "--openai-max-completion-tokens",
        type=int,
        default=None,
        help="Output cap (default 64, or 4096 with --openai-reasoning-effort)",
    )
    parser.add_argument(
        "--openai-endpoint",
        type=str,
        default=None,
        help="Optional OpenAI-compatible endpoint URL (use for self-hosted vLLM/GPT OSS)",
    )
    parser.add_argument(
        "--max-moves", type=int, default=None, help="Maximum number of moves to play"
    )

    args = parser.parse_args()
    main(args)
