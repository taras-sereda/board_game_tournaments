import argparse
import jax
import jax.numpy as jnp
import pgx
from pathlib import Path
env = pgx.make("chess")

# JIT-compile init and step
init = jax.jit(env.init)
step = jax.jit(env.step)

# Two "AIs" — here, both are random policies. Replace with your model.
def ai_black(rng, state):
    # Sample uniformly from the legal moves
    logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits)

def ai_white(rng, state):
    logits = jnp.log(state.legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits)

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # index by state.current_player
    players = [ai_white, ai_black]

    # Run one game
    rng = jax.random.PRNGKey(args.seed)
    rng, sub = jax.random.split(rng)
    state = init(sub)
    idx = 0
    while not state.terminated:
        rng, sub = jax.random.split(rng)
        # current_player is a 0/1 scalar telling whose turn it is
        action = players[int(state.current_player)](sub, state)
        state = step(state, action)
        if not args.no_save:
            state.save_svg(f"{out_dir}/{args.save_prefix}{idx:04}.svg")
        idx +=1

    # possible rewards:
    # [-1, 1] white wins, seed 0
    # [1, -1] black wins, seed 5
    # [0, 0] draw, seed 1, 2
    print(f"step {idx} rewards: {state.rewards}")

    # save the final position
    if not args.no_save:
        state.save_svg(f"{out_dir}/{args.save_prefix}final.svg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a single game of chess")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--save-prefix", type=str, default="state_", help="Prefix for saved SVGs")
    parser.add_argument("--out-dir", type=str, default="./states", help="Directory to save SVG files")
    parser.add_argument("--no-save", action="store_true", help="Do not save SVG files during play")
    args = parser.parse_args()
    main(args)