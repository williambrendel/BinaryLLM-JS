# Corpora for collision validation

This folder is for test corpora used to validate the token encoder's
collision rate against real prose in multiple languages.

Files here are **not** committed to git (see `.gitignore`); each developer
downloads them locally.

## Suggested corpora (Project Gutenberg, public domain)

These are well-formed prose with reasonable size (50K-200K words each):

### English
- *Alice's Adventures in Wonderland* (Carroll)
  https://www.gutenberg.org/files/11/11-0.txt
- *Pride and Prejudice* (Austen)
  https://www.gutenberg.org/files/1342/1342-0.txt

### French
- *Les Misérables, Tome I* (Hugo)
  https://www.gutenberg.org/files/17489/17489-0.txt
- *Madame Bovary* (Flaubert)
  https://www.gutenberg.org/cache/epub/14155/pg14155.txt

### Spanish
- *Don Quijote* (Cervantes)
  https://www.gutenberg.org/cache/epub/2000/pg2000.txt

### Code
- Some readable C++ project's source file — e.g. the SQLite amalgamation:
  https://www.sqlite.org/2024/sqlite-amalgamation-3450200.zip

## Quick download (all at once)

```bash
cd data/corpora
curl -L https://www.gutenberg.org/files/11/11-0.txt          -o alice_en.txt
curl -L https://www.gutenberg.org/files/1342/1342-0.txt      -o pride_en.txt
curl -L https://www.gutenberg.org/files/17489/17489-0.txt    -o miserables_fr.txt
curl -L https://www.gutenberg.org/cache/epub/14155/pg14155.txt -o bovary_fr.txt
curl -L https://www.gutenberg.org/cache/epub/2000/pg2000.txt -o quijote_es.txt
```

## Notes on UTF-8

Project Gutenberg files are UTF-8. The current v1 encoder only handles ASCII
letters and digits — any high-bit Unicode characters (accented letters like
`é`, `ñ`, `ü`) will currently be skipped. This means French and Spanish
collision rates will be artificially elevated because words like `école`
get encoded as `cole` (missing accent).

This is a known v1 limitation. Validating *current* collision rates gives
us a baseline; adding Unicode support in v2 should reduce the rate further.
