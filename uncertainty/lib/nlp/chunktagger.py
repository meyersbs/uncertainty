from nltk.chunk import ChunkParserI, tree2conlltags as to_tags
from nltk.corpus import treebank_chunk, conll2000
from nltk.tag import UnigramTagger, BigramTagger


def tag_chunks(chunk_sents):
    tag_sents = [to_tags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in chunk_tags] for chunk_tags in tag_sents]


CHUNKS = tag_chunks(treebank_chunk.chunked_sents()) + \
         tag_chunks(conll2000.chunked_sents())
TAGGER = BigramTagger(CHUNKS, backoff=UnigramTagger(CHUNKS))


class ChunkTagger(ChunkParserI):
    def parse(self, tokens):
        (tokens, tags) = zip(*tokens)
        chunks = TAGGER.tag(tags)
        return [(token, chunk[1]) for (token, chunk) in zip(tokens, chunks)]
