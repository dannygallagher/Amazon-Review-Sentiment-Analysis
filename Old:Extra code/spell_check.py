from autocorrect import Speller

check = Speller(lang = 'en')

def fixSentence(sentence):
    return check(sentence)
