#!/bin/sh
cargo bench --bench pomap2_bench -- memory 2>&1 | grep -v '^$'
