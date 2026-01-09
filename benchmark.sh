#!/bin/sh
cargo bench --color=always 2>&1 | grep '1.000 ('
