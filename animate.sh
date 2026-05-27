# Convert SVGs to PNGs
mkdir -p states/game_000/frames
for f in states/game_000/state_*.svg; do
    name=$(basename "$f" .svg)
    rsvg-convert "$f" -o "states/game_000/frames/${name}.png"
done

# Stitch into MP4
ffmpeg -framerate 1 -i states/game_000/frames/state_%04d.png \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       -c:v libx264 -pix_fmt yuv420p \
       states/game_000/game.mp4