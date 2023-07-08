#!/usr/bin/bash

rm -rf build
make html
cp -r build/html/* ./
git add *.html searchindex.js objects.inv _sources/*.txt source/*.rst _modules/index.html _modules/barmpy/*.html _static/*.css _static/*.js source/conf.py
