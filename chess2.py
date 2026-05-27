import argparse
import json
import jax
import pgx
 
from pathlib import Path
 
from util import ACTION_TO_UCI, state_to_fen
from player import AnthropicPlayer, OpenAIPlayer, RandomPlayer

env = pgx.make("chess")
init = jax.jit(env.init)
step = jax.jit(env.step)
 
def make_player(name, seed):
    if name == "anthropic":
        return AnthropicPlayer(model=args.anthropic_model)
    elif name == "openai":
        return OpenAIPlayer(model=args.openai_model)
    elif name == "gpt-oss":
        return OpenAIPlayer(
            model=args.openai_model,
            endpoint=args.openai_endpoint,
        )
    elif name == "random":
        return RandomPlayer(seed=seed)
    else:
        raise ValueError(f"Unknown player type: {name}")
 
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.key(args.seed)
    rng, sub = jax.random.split(rng)
    white_player = make_player(args.white, jax.random.key_data(sub)[0].item())
    rng, sub = jax.random.split(rng)
    black_player = make_player(args.black, jax.random.key_data(sub)[0].item())
    # Set up players: index 0 = white, index 1 = black
    players = [white_player, black_player]
 
    print(f"White: {white_player.name}")
    print(f"Black: {black_player.name}")
    print()

    state = init(sub)
    move_history: list[str] = []
    idx = 0
    max_moves = args.max_moves if args.max_moves is not None else int(1e3)
 
    while not state.terminated and idx < max_moves:
        fen = state_to_fen(state)
        current = int(state.current_player)
        color_name = "White" if current == 0 else "Black"
        print(f"Move {idx + 1} ({color_name} / {players[current].name}) — FEN: {fen}")
 
        action = players[current].choose_move(state, move_history)
        uci = ACTION_TO_UCI.get(action, "???")
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
        log = {
            "white": white_player.name,
            "black": black_player.name,
            "result": "white" if float(rewards[0]) == -1 else "black" if float(rewards[0]) == 1 else "draw",
            "moves": move_history,
            "final_fen": state_to_fen(state),
        }
        with open(f"{out_dir}/game_log.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nSaved {idx} SVGs + final + game_log.json to {out_dir}/")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM vs LLM chess via PGX")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--save-prefix", type=str, default="state_", help="Prefix for saved SVGs")
    parser.add_argument("--out-dir", type=str, default="./states", help="Directory to save SVG files")
    parser.add_argument("--no-save", action="store_true", help="Do not save SVG files during play")
 
    parser.add_argument("--white", type=str, default="anthropic",
                        choices=["anthropic", "openai", "gpt-oss", "random"],
                        help="Player for White (default: anthropic)")
    parser.add_argument("--black", type=str, default="openai",
                        choices=["anthropic", "openai", "gpt-oss", "random"],
                        help="Player for Black (default: openai)")
    parser.add_argument("--anthropic-model", type=str, default="claude-haiku-4-5",
                        help="Anthropic model name")
    parser.add_argument("--openai-model", type=str, default="gpt-5.4-nano",
                        help="OpenAI model name")
    parser.add_argument("--openai-endpoint", type=str, default=None,
                        help="Optional OpenAI-compatible endpoint URL (use for self-hosted vLLM/GPT OSS)")
    parser.add_argument("--max-moves", type=int, default=None, help="Maximum number of moves to play")
 
    args = parser.parse_args()
    main(args)
