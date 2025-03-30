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
  output_dir="$script_dir/tests/stream/testdata/detect_length"
  mkdir -p "$output_dir"

  _extract_frames "$output_dir" 0
}

extract_init_frames() {
  output_dir="$script_dir/tests/stream/testdata/extract_sectors"
  mkdir -p "$output_dir"

  _extract_frames "$output_dir" "-1"
}

rename() {
  exec 3>/dev/stdin
  while read -r filepath; do
  # echo "$filepath"
    filename="$(basename "$filepath")"
    name="${filename%.*}"
    echo -e "${Y}Start $filepath. Rename to?${E}"
    IFS= read -r newname <&3

    # substitute
    contains=$(grep -r --exclude-dir={.git,.idea,.venv,assets} --exclude='*.png' "$name" . || true)
    if [ -n "$contains" ]; then
      while read -r filepath2; do
        cmd="sed -i '' \"s/$name/$newname/g\" \"$filepath2\""
        echo -e "${G}Run command: ${cmd}${E}"
        eval "$cmd"
      done < <(echo "$contains" | awk -F : '{print $1}')
    else
      echo -e "${R}No contains: ${filename}${E}"
    fi

    # rename
    filepaths=$(find . -type d \( -path './.venv' -o -path './assets' -o -path './.git' \) -prune -o -name "$name.*" -print || true)
    if [ -n "$filepaths" ]; then
      while read -r filepath2; do
        newfilepath2="$(echo "$filepath2" | sed "s/$name/$newname/")"
        cmd="mv $filepath2 $newfilepath2"
        echo -e "${G}Run command: ${cmd}${E}"
        eval "$cmd"
      done < <(echo "$filepaths")
    else
      echo -e "${R}No filepaths: ${filename}${E}"
    fi

  done < <(find assets -type f \( -name '*.ts' -o -name '*.m4v' -o -name '*.mp4' \) | sort)
  exec 3>&-
}

run() {
  link="$1"
  streamlink --twitch-disable-ads --twitch-low-latency --stdout "$link" best | python main.py winner
}

run_vod() {
  link="$1"
  offset="${2:-00:00:00}"
  duration="${3:-00:03:00}"
  duration="${duration::-1}0"
  streamlink --hls-start-offset "$offset" --hls-duration "$duration" --stdout "$link" best | python main.py winner
}

meme() {
  # Origin
  #curl 'https://memealerts.com/api/user/balance' \
  #  -H 'accept: */*' \
  #  -H 'accept-language: en-US,en;q=0.9,ru;q=0.8' \
  #  -H 'authorization: Bearer foo_bearer' \
  #  -H 'cache-control: no-cache' \
  #  -H 'content-type: application/json' \
  #  -b 'lang=ru; theme=dark; _csrf=foo_csrf; x-csrf-token=foo_x_csrf' \
  #  -H 'origin: https://memealerts.com' \
  #  -H 'pragma: no-cache' \
  #  -H 'priority: u=1, i' \
  #  -H 'referer: https://memealerts.com/twitch-channel' \
  #  -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
  #  -H 'sec-ch-ua-mobile: ?0' \
  #  -H 'sec-ch-ua-platform: "macOS"' \
  #  -H 'sec-fetch-dest: empty' \
  #  -H 'sec-fetch-mode: cors' \
  #  -H 'sec-fetch-site: same-origin' \
  #  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36' \
  #  -H 'x-csrf-token: foo_x_csrf' \
  #  --data-raw '{"username":"twitch-channel"}'

  sticker_id="65f1e02fbc597ae824b07603"
  max_length=50

  bearer="Bearer $1"
  streamer_link="$2"
  my_name="$3"
  message="$4"

  length=$(echo -n "$message" | wc -m | awk '{print $1}')
  [ "$length" -gt "$max_length" ] && echo "The length is $length. The max is $max_length." && exit 1

  # Grab streamer id
  streamer_id=$(curl -s 'https://memealerts.com/api/supporters/me' -H "authorization: $bearer" \
    | jq -r ".data.[]
      | select((.streamerLink == \"$streamer_link\") or (.streamerName | ascii_downcase == (\"$streamer_link\" | ascii_downcase)))
      | .streamerId")
  [ -z "$streamer_id" ] && echo "streamer id not found" && exit 1

  # Grab x-csrf-token and _csrf
  cookies=$(curl -s -D - 'https://memealerts.com/api/user/current' -H "authorization: $bearer" \
    | grep -i "Set-Cookie" | awk -F': ' '{print $2}' | tr -d '\r')
  csrf_token=$(echo "$cookies" | grep -o 'x-csrf-token=[^;]*' | cut -d= -f2)
  csrf=$(echo "$cookies" | grep -o '_csrf=[^;]*' | cut -d= -f2)
  [ -z "$csrf_token" ] && echo "no csrf token" && exit 1
  [ -z "$csrf" ] && echo "no _csrf" && exit 1

  # Send meme
  curl 'https://memealerts.com/api/sticker/send' \
    -H "authorization: $bearer" \
    -H 'content-type: application/json' \
    -b "lang=ru; theme=dark; _csrf=$csrf; x-csrf-token=$csrf_token" \
    -H "x-csrf-token: $csrf_token" \
    --data-raw "{\"toChannel\":\"$streamer_id\",\"stickerId\":\"$sticker_id\",\"isSoundOnly\":true,\"topic\":\"Favorites\",\"name\":\"$my_name\",\"isMemePartyActive\":false,\"message\":\"$message\"}"
}

[[ $# -lt 1 ]] && echo "No function passed" && exit 1
fn="$1"
shift
"$fn" "$@"
