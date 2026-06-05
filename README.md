
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

# experiment Athropic vs OpenAI $13 each.
python chess.py \
  --out-dir states/game_017 \
  --white anthropic \
  --black openai \
  --anthropic-model claude-opus-4-8 \
  --anthropic-adaptive-thinking \
  --anthropic-effort high \
  --anthropic-max-tokens 128000 \
  --openai-model gpt-5.5 \
  --openai-reasoning-effort high \
  --openai-max-completion-tokens 128000
```

### Games:
1. `game_15` GPT-OSS vs GPT-OSS, seed=52, draw.
2. `game_10` GPT-OSS vs GPT-OSS, seed=52, white.
3. `game_17` claude-opus-4-8 vs gpt-5.5

