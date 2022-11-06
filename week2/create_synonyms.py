import fasttext
model = fasttext.load_model("/workspace/datasets/fasttext/title_model.bin")
TOP_WORDS_DIR = "/workspace/datasets/fasttext/top_words.txt"
SYNONYMS_CSV_DIR = "/workspace/datasets/fasttext/synonyms.csv"
THRESHOLD = 0.75

with open(TOP_WORDS_DIR, "r") as top_words, open(SYNONYMS_CSV_DIR, "w") as synonyms_csv:
    for word in top_words:
        word = word.strip()
        nn = model.get_nearest_neighbors(word)
        n_words = [synonym.strip() for (sim_coef, synonym) in nn if (sim_coef >= THRESHOLD and synonym.strip() != word)]
        if len(n_words) > 0:
            synonyms_csv.write(f"{word},{','.join(n_words)}")
            synonyms_csv.write("\n")
