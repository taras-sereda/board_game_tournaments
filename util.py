import re
import pgx.experimental.chess as ec
import pgx._src.games.chess as chess_core
import jax.numpy as jnp

UNC_ACTION = "???"


def _build_action_uci_maps():
    """Build bidirectional mappings between PGX action indices and UCI move strings."""
    promo_chars = {0: "r", 1: "b", 2: "n"}

    def sq_to_uci(sq):
        return chr(ord("a") + sq // 8) + str(sq % 8 + 1)

    action_to_uci = {}
    uci_to_action = {}

    for action_id in range(4672):
        from_sq = action_id // 73
        plane = action_id % 73
        to_sq = chess_core.FROM_PLANE[from_sq, plane]
        if to_sq == -1:
            continue

        from_str = sq_to_uci(from_sq)
        to_str = sq_to_uci(to_sq)

        if plane < 9:
            uci = from_str + to_str + promo_chars[plane // 3]
        elif from_sq % 8 == 6 and to_sq % 8 == 7:
            uci = from_str + to_str + "q"
        else:
            uci = from_str + to_str

        action_to_uci[action_id] = uci
        uci_to_action[uci] = action_id

    return action_to_uci, uci_to_action


ACTION_TO_UCI, UCI_TO_ACTION = _build_action_uci_maps()


def _black_to_move(state) -> bool:
    return state_to_fen(state).split()[1] == "b"


def _flip_uci(uci: str) -> str:
    # PGX encodes actions in the mover's frame (vertical mirror: file kept,
    # rank -> 9-rank). Only the square coords flip; promo suffix is unchanged.
    return uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]


def action_to_uci(action: int, state) -> str:
    """Absolute (real-board) UCI for an action id, given whose turn it is."""
    uci = ACTION_TO_UCI.get(int(action))
    if uci is None:
        return UNC_ACTION
    return _flip_uci(uci) if _black_to_move(state) else uci


def get_legal_uci_moves(state):
    flip = _black_to_move(state)
    legal = jnp.where(state.legal_action_mask)[0]
    out = []
    for a in legal:
        u = ACTION_TO_UCI.get(int(a))
        if u is not None:
            out.append(_flip_uci(u) if flip else u)
    return out


def uci_to_action_id(uci_move: str, state) -> int | None:
    """Convert a UCI string to a PGX action index, checking legality.

    Handles cases like:
    - Exact match: 'e2e4'
    - Queen promotion without suffix: 'e7e8' -> tries 'e7e8q'
    - With or without promotion suffix
    """
    uci_move = uci_move.strip().lower()
    if _black_to_move(state):
        uci_move = _flip_uci(uci_move)  # real -> mover-frame before lookup

    # Direct lookup
    if uci_move in UCI_TO_ACTION:
        aid = UCI_TO_ACTION[uci_move]
        if state.legal_action_mask[aid]:
            return aid

    # Try adding queen promotion suffix
    if len(uci_move) == 4:
        with_q = uci_move + "q"
        if with_q in UCI_TO_ACTION:
            aid = UCI_TO_ACTION[with_q]
            if state.legal_action_mask[aid]:
                return aid

    # Try stripping promotion suffix (maybe LLM added one unnecessarily)
    if len(uci_move) == 5:
        without = uci_move[:4]
        if without in UCI_TO_ACTION:
            aid = UCI_TO_ACTION[without]
            if state.legal_action_mask[aid]:
                return aid

    return None


def state_to_fen(state) -> str:
    return ec.to_fen(state)


SYSTEM_PROMPT = """\
You are a chess engine. You will be given a board position as a FEN string and a list of legal moves in UCI notation. Your job is to choose the best move.

Rules:
- Respond with ONLY a single UCI move (e.g. e2e4, g1f3, e7e8q for queen promotion).
- The move MUST be from the provided list of legal moves.
- Do NOT include any explanation, commentary, or extra text.
- Just the move, nothing else."""


def build_user_prompt(
    fen: str, legal_moves: list[str], color: str, move_history: list[str]
) -> str:
    parts = [f"Position (FEN): {fen}"]
    if move_history:
        parts.append(f"Move history: {' '.join(move_history)}")
    parts.append(f"You are playing as {color}.")
    parts.append(f"Legal moves: {', '.join(legal_moves)}")
    parts.append("Your move:")
    return "\n".join(parts)


def parse_uci_from_response(text: str) -> str | None:
    """Extract a UCI move from LLM response text."""
    text = text.strip()
    # Try the whole response as a move
    match = re.match(r"^([a-h][1-8][a-h][1-8][qrbn]?)$", text)
    if match:
        return match.group(1)
    # Try to find a move anywhere in the response
    match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", text)
    if match:
        return match.group(1)
    return None
