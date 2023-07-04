import spacy
from spacy.lang.en import English
from spacy.attrs import ORTH, NORM
import re

nlp = English()
tokenizer = nlp.tokenizer
tokenizer.add_special_case("etc.", [{ORTH: "etc."}])
tokenizer.add_special_case("``", [{ORTH: "``"}])
tokenizer.add_special_case("no.", [{ORTH: "no."}])
tokenizer.add_special_case("No.", [{ORTH: "No."}])
tokenizer.add_special_case("%pw", [{ORTH: "%pw"}])
tokenizer.add_special_case("%PW", [{ORTH: "%PW"}])
tokenizer.add_special_case("mr.", [{ORTH: "mr."}])
tokenizer.add_special_case("goin'", [{ORTH: "goin"}, {ORTH: "'"}])
tokenizer.add_special_case("'cause", [{ORTH: "'"}, {ORTH: "cause"}])
tokenizer.add_special_case("'m", [{ORTH: "'m"}])
tokenizer.add_special_case("'ve", [{ORTH: "'ve"}])
'''
tokenizer.add_special_case("dont", [{ORTH: "dont"}])
tokenizer.add_special_case("doesnt", [{ORTH: "doesnt"}])
tokenizer.add_special_case("cant", [{ORTH: "cant"}])
tokenizer.add_special_case("havent", [{ORTH: "havent"}])
tokenizer.add_special_case("didnt", [{ORTH: "didnt"}])
tokenizer.add_special_case("youre", [{ORTH: "youre"}])
tokenizer.add_special_case("wont", [{ORTH: "wont"}])
tokenizer.add_special_case("im", [{ORTH: "im"}])
tokenizer.add_special_case("aint", [{ORTH: "aint"}])
'''

tokenizer.add_special_case("<<", [{ORTH: "<<"}])
tokenizer.add_special_case(">>", [{ORTH: ">>"}])


SPECIAL_CASES = {
    'wo': 'will',
    'Wo': 'Will',
    'ca': 'can',
    'Ca': 'Can'
}


def tokenize(text):
    # Add spaces for some cases
    text = re.sub(r'([-:\ / $])', r' \1 ', text)
    # Split dates, e.g. "11th" to "11 th"
    text = re.sub(r'(\d+)(st|nd|rd|th|s|am|pm)', r'\1 \2', text)
    # Remove double spaces
    text = re.sub(r'\s+', r' ', text)

    # Tokenize the text
    tokens = tokenizer(text)
    tokens = fix_special_cases([t.text for t in tokens])

    return ' '.join(tokens)

def fix_special_cases(tokens, nltk=False):
    for i in range(len(tokens)):
        if tokens[i] == "n't" and tokens[i-1] in SPECIAL_CASES:
            tokens[i-1] = SPECIAL_CASES[tokens[i-1]]
        if nltk:
            if tokens[i] in {"``", "''"}:
                tokens[i] = '"'
    return tokens