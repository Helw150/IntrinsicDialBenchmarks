#!/usr/bin/env bash
# Download all words in a category from wiktionary
# Usage: ./download-wiktionary-category.sh "English_contractions" > contractions.txt

set -euo pipefail
IFS=$'\n\t'

download() {
    local category="$1"
    local continue="${2:-}"
    if [[ $continue == "" ]]; then
	json[$category]='["'
    fi
    local resp="$(curl --silent --data-urlencode cmtitle=Category:${category} "https://en.wiktionary.org/w/api.php?action=query&list=categorymembers&cmprop=title&cmlimit=max&cmcontinue=${continue}&format=json")"
    local subcat="$(echo $resp | jq -r '.query.categorymembers[] | select (.ns > 0) | .title' | cut -d ":" -f2 | sed 's/ /_/g')"
    local next="$(echo "$resp" | jq -r '.continue.cmcontinue')"
    for CAT in $subcat
    do
        download $CAT
    done
    json[$category]+="$(echo $resp | jq -r '.query.categorymembers[] | select (.ns == 0) | .title' | sed -z 's/\n/\","/g')"
    if [[ $next != null ]]; then
        download "$category" "$next"
    else
	i=$category
	json[$category]=$(echo "${json[$i]}" | head -c -3)
	json[$category]+=]
    fi
    
}
declare -A json
download "$1"
tmp={
for i in "${!json[@]}"
do
    tmp+="\"$i\":${json[$i]},"
done
json=$(echo $tmp | head -c -2)
json+=}
echo $json > tmp
# Remove lines that aren't relevant and cause issues to JSON Parsing
cat tmp | sed 's/],/],\n/g' | grep -v ':],' | grep -v 'Hebrew' > loanwords.json
rm tmp
