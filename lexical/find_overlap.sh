jq ".[]" loanwords.json | grep '"' | sed "s/[ ,]//g" > loanwords_flat.txt
cat wiktionary_indian_english_lexicon_quiz.json | jq ".term" > test_flat.txt
comm -12 <(cat loanwords_flat.txt | sort) <(cat test_flat.txt | sort) | sed "s/\"//g" > overlap.txt
rm test_flat.txt
rm loanwords_flat.txt
cat overlap.txt 
