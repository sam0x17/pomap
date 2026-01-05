#!/bin/sh
cargo bench --color=always --features bench-string |& grep --color=never '1.000 ('

