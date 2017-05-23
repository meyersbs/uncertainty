from . import chunktagger, lemmatizer, postagger, stemmer, tokenizer


class Summarizer(object):
    def __init__(self, text):
        self.text = text

    def execute(self):
        tokens = tokenizer.NLTKTokenizer(self.text).execute()
        stems = stemmer.Stemmer(tokens).execute()
        pos = postagger.PosTagger(tokens).execute()
        chunk = chunktagger.ChunkTagger().parse(pos)

        summary = zip(tokens, stems, pos, chunk)
        return summary
