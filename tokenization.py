# building a tokenizer for the data
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace

def build_tokenizer_for_word_embeddings(vocab):
	# word level tokenizer
	# input: vocab
	# output: tokenizer
	model = WordLevel(vocab, "[UNK]")
	tokenizer = Tokenizer(model)
	# unicode normalization, lowercasing, and accent removal
	tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
	tokenizer.pre_tokenizer = Whitespace()
	return tokenizer
