LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


class Text:
    def __init__(self, text, nlp):
        self.tokens = nlp(text)
        self.lemmas = [token.lemma_.lower() for token in self.tokens
                       if token.is_alpha and not token.is_stop]

    def count_mean_word_len(self):
        mean_word_len = 0
        n_words = 0

        for token in self.tokens:
            if token.is_alpha:
                mean_word_len += len(token.text)
                n_words += 1

        mean_word_len /= n_words
        return mean_word_len

    def count_mean_sentence_len(self):
        mean_sentence_len = 0
        n_sentences = 0

        for sentence in self.tokens.sents:
            for token in sentence:
                if token.is_alpha:
                    mean_sentence_len += 1
            n_sentences += 1

        return mean_sentence_len / n_sentences

    def get_words_from_level_lists(self, word2level):
        level_words = {level: set() for level in LEVELS}

        for lemma in self.lemmas:
            level = word2level.get(lemma)
            if level:
                level_words[level].add(lemma)

        for level, words in level_words.items():
            level_words[level] = sorted(words)

        return level_words
