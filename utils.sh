#!/usr/bin/env bash

set -euo pipefail

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
