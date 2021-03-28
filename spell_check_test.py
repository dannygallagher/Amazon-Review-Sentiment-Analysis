from spellchecker import SpellChecker

spell = SpellChecker()

sentence = ['The', 'quick', 'bronw', 'fox', 'jumps', 'oevr', 'the', 'lazzy', 'dog']

mispelled = spell.unknown(sentence)

for word in mispelled:
    print(spell.correction(word))
