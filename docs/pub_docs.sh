#!/usr/bin/bash

rm -rf build
make html
cp -r build/html/* ./
git add *.html searchindex.js objects.inv _sources/*.txt source/*.rst
