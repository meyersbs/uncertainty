from nltk.chunk import ChunkParserI
from nltk.chunk import conllstr2tree as to_tree
from nltk.chunk import tree2conlltags as to_tags
from nltk.corpus import treebank_chunk, conll2000
from nltk.tag import UnigramTagger, BigramTagger

def tag_chunks(chunk_sents):
    tag_sents = [to_tags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in chunk_tags] for chunk_tags in tag_sents]


class SuperChunker(ChunkParserI):
    def __init__(self):
        self._chunks = tag_chunks(treebank_chunk.chunked_sents())
        self._chunks += tag_chunks(conll2000.chunked_sents())
        self._backoff = UnigramTagger(self._chunks)
        self._chunk_tagger = BigramTagger(self._chunks, backoff=self._backoff)

    def parse(self, sent_list):
        (tokens, tags) = zip(*sent_list)
        chunks = self._chunk_tagger.tag(tags)
        tok_pos_chunk = zip(tokens, chunks)
        lines = [{"token": t, "pos": p, "chunk": c} for (t, (p, c)) in tok_pos_chunk if c]
        return lines

#chunker = TreebankChunker()
#print(chunker.parse(pos_tag(["Cells", "in", "Regulating", "Cellular", "Immunity"])))
#print(chunker.parse(pos_tag(["I", "am", "the", "walrus", "."])))
