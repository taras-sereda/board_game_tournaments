
### Usage

```bash
python chess.py \
    --out-dir ./states/game_005 \
    --seed 42 \
    --openai-model  openai/gpt-oss-20b \
    --black gpt-oss \
    --white gpt-oss \

# add self-serving end-point address
#    --openai-endpoint http://{host}:{port}/v1

# opus4.8 vs gpt5.5
# thinking mode high
python chess.py \
  --out-dir states/game_014 \
  --white anthropic \
  --black openai \
  --anthropic-model claude-opus-4-8 \
  --anthropic-adaptive-thinking \
  --anthropic-effort high \
  --anthropic-max-tokens 4096 \
  --openai-model gpt-5.5 \
  --openai-reasoning-effort high \
  --openai-max-completion-tokens 4096
```

