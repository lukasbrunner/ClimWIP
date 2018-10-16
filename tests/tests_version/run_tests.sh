#!/bin/bash

case $1 in
    'reference')
        # if ! [ -z "$(ls -A data/reference/)" ]; then
        #     echo "Delting existing files..."
        #     rm -r data/reference/*  # clean up first
        #     echo "Delting existing files... DONE"
        # fi
        ./../../model_weighting.py test_reference -f configs/config_test_versions
        ;;
    'new')
        if ! [ -z "$(ls -A data/new/)" ]; then
            echo "Delting existing files..."
            rm -r data/new/*  # cean up first
            echo "Delting existing files... DONE"
        fi
        ./../../model_weighting.py test_new -f configs/config_test_versions
        ./test_output_change.py
        ;;
    *)
        echo 'Should be one of [reference | new]'
esac
