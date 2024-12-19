#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")
assets_dir="$script_dir/assets"
G='\033[0;32m'
LG='\033[1;32m'
Y='\033[1;33m'
R='\033[0;31m'
E='\033[0m'

binary_search() {
  dir_path="$1"
  extra_check="$2"
  low=1
  high=$(ls -A "$dir_path" | wc -l | awk '{print $1}')

  stty -echo -icanon
  while [[ $low -le $high ]]; do
    mid=$(((low + high) / 2))
    mid_image=$(printf "frame%04d.png" $mid)
    open -R "$dir_path/$mid_image"
    osascript -e 'tell application "iTerm" to activate'

    echo -e "$mid_image was open.\nIs the image left or right from the target? (← or →, f — found, a — again, s — skip)"
    input=$(dd bs=1 count=1 2>/dev/null)
    if [[ "$input" == $'\x1b' ]]; then
      input2=$(dd bs=1 count=2 2>/dev/null)

      # if left
      if [[ "$input2" == "[D" ]]; then
        # (finding the end of spin)
        if [[ "$extra_check" == "behind" ]]; then
          mid2=$((mid - 1))
          mid_image2=$(printf "frame%04d.png" $mid2)
          echo -e "${Y}Lookbehind check. $mid_image2 was open.\nIs the image left or right from the target? (← or →)${E}"
          open -R "$dir_path/$(printf "frame%04d.png" $mid2)"
          osascript -e 'tell application "iTerm" to activate'
          input3=$(dd bs=1 count=1 2>/dev/null)
          [[ "$input3" != $'\x1b' ]] && echo -e "${R}Invalid input.${E}" && continue
          input4=$(dd bs=1 count=2 2>/dev/null)
          [[ "$input4" == "[C" ]] && break
          [[ "$input4" != "[D" ]] && echo -e "${R}Invalid input.${E}" && continue
        fi
        high=$((mid - 1)) && continue
      fi

      # if right
      if [[ "$input2" == "[C" ]]; then
        # (finding the init of spin)
        if [[ "$extra_check" == "ahead" ]]; then
          mid2=$((mid + 1))
          mid_image2=$(printf "frame%04d.png" $mid2)
          echo -e "${Y}Lookahead check. $mid_image2 was open.\nIs spin frame? (← or →)${E}"
          open -R "$dir_path/$(printf "frame%04d.png" $mid2)"
          osascript -e 'tell application "iTerm" to activate'
          input3=$(dd bs=1 count=1 2>/dev/null)
          [[ "$input3" != $'\x1b' ]] && echo -e "${R}Invalid input.${E}" && continue
          input4=$(dd bs=1 count=2 2>/dev/null)
          [[ "$input4" == "[D" ]] && break
          [[ "$input4" != "[C" ]] && echo -e "${R}Invalid input.${E}" && continue
        fi
        low=$((mid + 1)) && continue
      fi

      echo -e "${R}Invalid input.${E}" && continue
    fi
    [[ "$input" == "f" ]] && echo -e "${R}Found.${E}" && RESULT="$mid" && return 0
    [[ "$input" == "a" ]] && echo -e "${R}Run again.${E}" && return 1
    [[ "$input" == "s" ]] && echo -e "${R}Skipped.${E}" && RESULT=null && return 0

    echo -e "${R}Invalid input.${E}"
  done

  echo -e "${G}Found $mid_image ${E}"
  RESULT="$mid"

  echo -e "${LG}Final check. $mid_image was open. Is it? (y or n)${E}"
  open -R "$dir_path/$mid_image"
  input=$(dd bs=1 count=1 2>/dev/null)
  [[ "$input" != "y" ]] && echo -e "${R}Restart!${E}" && return 1

  stty echo icanon
}

manual_finder() {
  exec 3< <(find "$assets_dir" -type d | awk -F'/' '{print NF, $0}' | sort -nr | awk 'NR==1 {max=$1} $1==max {print $2}')
  while read -r dir_path <&3; do
    filepath="assets${dir_path#*/assets}"
    if jq -r '.[].path' < "$script_dir/output.json" | grep -q "$filepath"; then
      echo -e "${R}Skipped $dir_path.${E}"
      continue
    fi
    echo -e "${G}Start $dir_path.${E}"

    echo -e "${LG}Find first wheel frame.${E}"
    while ! binary_search "$dir_path" ""; do
      :
    done
    first_wheel_frame="$RESULT"

    echo -e "${LG}Find init spin frame.${E}"
    while ! binary_search "$dir_path" "ahead"; do
      :
    done
    init_frame="$RESULT"

    echo -e "${LG}Find end spin frame.${E}"
    while ! binary_search "$dir_path" "behind"; do
      :
    done
    end_spin_frame="$RESULT"

    json="{\"path\": \"$filepath\", \"first_wheel_frame\": $first_wheel_frame, \"init_frame\": $init_frame, \"end_spin_frame\": $end_spin_frame}"
    echo "$json" | jq .

    jq ". += [$json]" "$script_dir/output.json" > "$script_dir/tmp.json" && mv "$script_dir/tmp.json" "$script_dir/output.json"
  done
  exec 3<&-
}

cut() {
  while read -r filepath; do
    echo -e "${G}Start $filepath${E}"
    dir="${filepath%.*}"
    mkdir -p "$dir"
    ffmpeg -nostdin -i "$filepath" -qscale:v 31 -vf "scale=iw*0.1:ih*0.1" "$dir/frame%04d.png"
  done < <(find "$assets_dir" -type f \( -iname \*.ts -o -iname \*.mp4 -o -iname \*.m4v \))
}

set_length() {
  exec 3< <(jq -r '.[].path' "$script_dir/output.json")

  while read -r dir <&3; do
    echo -e "${G}Start $dir${E}"
    filepath=$(find "$assets_dir" -type f -name "$(basename "$dir").*" | tail -1)
    open "$filepath"
    read -rp "length: " length
    filepath="assets${filepath#*/assets}"
    json="{\"path\": \"$dir\", \"length\": $length, \"filepath\": \"$filepath\"}"
    echo "$json" | jq .

    jq ". += [$json]" "$script_dir/length.json" > "$script_dir/tmp.json" && mv "$script_dir/tmp.json" "$script_dir/length.json"
    ps aux | grep -v grep | grep /Applications/IINA.app/Contents/MacOS/IINA | awk '{print $2}' | xargs kill -9
  done
}

time_diff() {
  start_time="$1"
  end_time="$2"

  start_seconds=$(echo "$start_time" | awk -F: '{print ($1 * 3600) + ($2 * 60) + $3}')
  end_seconds=$(echo "$end_time" | awk -F: '{print ($1 * 3600) + ($2 * 60) + $3}')
  diff_seconds=$((end_seconds - start_seconds))

  hours=$((diff_seconds / 3600))
  minutes=$(((diff_seconds % 3600) / 60))
  seconds=$((diff_seconds % 60))

  printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}

scrap_external() {
  while read -r channel link start end; do
    dir="$assets_dir/$channel"
    video_id=$(basename "$link")
    start_escaped=$(echo "$start" | tr : _)
    filepath="${dir}/${video_id}_${start_escaped}.ts"
    duration=$(time_diff "$start" "$end")

    if [ -f "$filepath" ]; then
      echo "Skipped $link ($start)"
      continue
    fi

    echo "Start download $link"
    streamlink --hls-start-offset "$start" --hls-duration "$duration" "$link" best -o "$filepath"
  done < <(jq -r '.[] | "\(.channel) \(.link) \(.start) \(.end)"' "$script_dir/links.json")
}

_extract_frames() {
  output_dir="$1"
  init_frame_offset="$2"

  # dump frames
  data="["
  while read -r filepath init_frame length; do
    [ "$init_frame" == "null" ] && continue
    echo -e "${G}Start ${filepath}${E}"
    frame_index=$((init_frame + init_frame_offset))
    filename="$(basename "${filepath%.*}").png"
    output_path="$output_dir/$filename"
    ffmpeg -nostdin -i "$script_dir/$filepath" -vf "select=eq(n\,${frame_index})" -vsync vfr -q:v 1 "$output_path"

    data+="{\"path\": \"$filename\"},"
  done < <(jq -r '.[] | "\(.filepath) \(.init_frame) \(.length)"' "$script_dir/input.json")

  data="${data%,}]"
  echo "$data" | jq -r > "$output_dir/input.json"
}

extract_length_frames() {
  output_dir="$script_dir/detect_length"
  mkdir -p "$output_dir"

  _extract_frames "$output_dir" 0
}

extract_init_frames() {
  output_dir="$script_dir/extract_sectors"
  mkdir -p "$output_dir"

  _extract_frames "$output_dir" "-1"
}

[[ $# -lt 1 ]] && echo "No function was passed" && exit 1
fn="$1"
shift
"$fn" "$@"
