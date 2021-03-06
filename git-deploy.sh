#!/bin/bash

echo "Publishing the website ..."

echo "Building the website ..."
bundle exec jekyll build 

echo "Git add ..."
git add .

echo "Committing ..."
read -r -p 'Commit message: ' desc
git commit -m "$desc"

echo "Pushing the changes to gh-pages branch ..."
git push -u origin gh-pages

# Backup commands
# bundle update github-pages
# JEKYLL_ENV=production jekyll build