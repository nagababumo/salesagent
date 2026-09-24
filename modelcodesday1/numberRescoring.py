from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
    InverseNormalizer,
)

itn = InverseNormalizer(
    lang="hi",
    input_case="lower_cased",
)

tests = [
    "पाँच हजार",
    "दस हजार",
    "एक लाख",
    "एक लाख पच्चीस हजार",
    "पाँच सौ",
    "दो करोड़",
]

for text in tests:
    result = itn.inverse_normalize(text, verbose=False)
    print(f"{text}  -->  {result}")
    
    
    

tn = Normalizer(
    lang="hi",
    input_case="cased",
)

tests = [
    "5000",
    "10000",
    "125000",
    "500",
    "20000000",
]

for text in tests:
    result = tn.normalize(text, verbose=False)
    print(f"{text}  -->  {result}")    