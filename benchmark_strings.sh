#!/bin/sh
cargo bench --color=always --features bench-string 2>&1 | grep '1.000 ('
