#!/bin/bash

wget -nc https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.json

# Define an array of URLs
urls=(
    "https://en.wiktionary.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Bangladeshi_English&gcmlimit=max&format=json"

    "https://en.wiktionary.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Pakistani_English&gcmlimit=max&format=json"

    "https://en.wiktionary.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Indian_English&gcmlimit=max&format=json"
      "https://en.wiktionary.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Indian_English&gcmlimit=max&gcmcontinue=page|47414e44550a47414e4455|6344566&format=json"
  "https://en.wiktionary.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Indian_English&gcmlimit=max&gcmcontinue=page|4e52430a4e5243|184888&format=json" "https://en.wiktionary.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:Indian_English&gcmlimit=max&gcmcontinue=page|554e495155454944454e54494649434154494f4e4e554d4245520a554e495155452049444544454e54494649434154494f4e204e554d424552|2915079&format=json"
)

rm output.csv
# Iterate over the URLs using a for loop
for url in "${urls[@]}"; do
  # Your logic here, e.g., using the URL in some command
  echo "Processing URL: $url"
  wget -O - -q "$url"  | jq -c -r '.query.pages | .[]' | grep -v Category | jq ".title" >> output.csv
  # Add your command using "$url" here
done

if ! test -f grep_v.csv; then
    cat output.csv | awk '{print "\"word\": " $0", \"lang\": \"English\""}' > grep_v.csv
fi

if ! test -f indian_english_subset.json; then
    grep -f grep_v.csv kaikki.org-dictionary-English.json > indian_english_subset.json
fi

if ! test -f filtered_ss.json; then
    cat indian_english_subset.json | grep -v -E "archaic|obsolete|historical" > filtered_ss.json
fi

rm benchmark.json


