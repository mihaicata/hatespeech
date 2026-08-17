"""Keyword/lexicon list for the keyword-matching method.

Deliberately a moderate, illustrative set (insults, profanity, threats,
dehumanizing language) rather than an exhaustive slur dictionary — a real
production system would plug in a maintained lexicon (e.g. HateBase) here
instead. English and German share one flat set since matching is just
substring/word lookup, no per-language logic needed.
"""

KEYWORDS = {
    # insults / dehumanization (EN)
    "idiot", "moron", "stupid", "dumb", "loser", "pathetic", "worthless",
    "garbage", "trash", "scum", "subhuman", "disgusting", "vermin", "filth",
    "retard", "freak", "ugly", "brainless", "clown",
    # profanity / vulgarity (EN)
    "fuck", "fucking", "shit", "bullshit", "bastard", "bitch", "asshole",
    "piss off", "screw you",
    # threats / violence (EN)
    "kill you", "hope you die", "should die", "kys", "beat you up",
    "hunt you down", "burn in hell", "you deserve to die",
    # insults / dehumanization (DE)
    "idiot", "trottel", "dumm", "blöd", "bekloppt", "widerlich", "abschaum",
    "dreck", "wertlos", "hässlich", "vollidiot", "spast", "missgeburt",
    "untermensch",
    # profanity / vulgarity (DE)
    "scheisse", "scheiße", "verpiss dich", "arschloch", "hurensohn",
    "wichser", "fotze", "schlampe",
    # threats / violence (DE)
    "bring dich um", "du sollst sterben", "verrecken", "geh sterben",
}

# Cheap leetspeak / obfuscation normalization so "h4te", "$hit" etc. still match.
LEET_MAP = str.maketrans({
    "4": "a", "0": "o", "1": "i", "3": "e", "5": "s", "7": "t",
    "@": "a", "$": "s", "+": "t",
})
