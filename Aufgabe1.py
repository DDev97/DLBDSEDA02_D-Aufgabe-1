from spellchecker import SpellChecker

spell = SpellChecker(language='de')

falsch = spell.unknown(['manchmal', 'hier', 'woerter'])

for word in falsch:
    print(spell.correction(word))