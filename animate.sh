# Convert SVGs to PNGs

dir_name=$1
mkdir -p ${dir_name}/frames
for f in ${dir_name}/state_*.svg; do
    name=$(basename "$f" .svg)
    rsvg-convert "$f" -o "${dir_name}/frames/${name}.png"
done

# Stitch into MP4
ffmpeg -framerate 1 -i ${dir_name}/frames/state_%04d.png \
       -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
       -c:v libx264 -pix_fmt yuv420p \
       ${dir_name}/game.mp4