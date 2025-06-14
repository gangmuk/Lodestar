package tokenizer

import (
	"strings"
)

type wordTokenizer struct{}

func NewWordTokenizer() Tokenizer {
	return &wordTokenizer{}
}

func (w wordTokenizer) TokenizeInputText(text string) ([]byte, error) {
	// Split text by whitespace and count words
	words := strings.Fields(text)

	// Return a byte slice with length equal to word count
	// Since only len(tokens) is used, we can just make a slice of the right size
	tokens := make([]byte, len(words))

	return tokens, nil
}
