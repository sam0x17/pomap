#!/bin/sh
cargo bench --color=always 2>&1 \
  | grep --line-buffered -E 'rank' \
  | grep --line-buffered 'pomap' \
  | awk '{
      if ($0 ~ /^ *1st/) printf "\033[32m%s\033[0m\n", $0;
      else printf "\033[31m%s\033[0m\n", $0;
      fflush();
    }'
