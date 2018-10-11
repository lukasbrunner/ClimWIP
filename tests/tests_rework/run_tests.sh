#!/bin/bash

coverage run -p test_rework_model_weighting.py
coverage report -m --rcfile=.coveragerc.ini
