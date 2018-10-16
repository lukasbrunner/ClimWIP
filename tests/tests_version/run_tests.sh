#!/bin/bash

case $1 in
    'reference')
        if [ -z "$(ls -A data/reference/)" ]; then
            rm -r data/reference/*  # clean up first
        fi
        ./../../model_weighting.py test_reference
        ;;
    'new')
        if [ -z "$(ls -A data/new/)" ]; then
            rm -r data/new/*  # cean up first
        fi
        ./../../model_weighting.py test_new
        ./test_output_change.py
        ;;
    *)
        echo 'Should be one of [reference | new]'
esac
