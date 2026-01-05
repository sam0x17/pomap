#!/bin/sh
cargo bench --color=always |& grep --color=never '1.000 ('

