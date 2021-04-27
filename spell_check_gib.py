!apt install -qq enchant
!pip install pyenchant

import enchant
from enchant.checker import SpellChecker

from autocorrect import Speller
import dask
import dask.dataframe as dd


dask_df = dd.from_pandas(df, npartitions = 16)

def detect_gib(sentence):
  len_sent = len(sentence.split())
  chkr = SpellChecker("en_US")

  chkr.set_text(sentence)
  num_errors = 0
  for err in chkr:
    num_errors = num_errors + 1
  if num_errors > len_sent / 2:
    return None
  else:
    return sentence

dask_df['reviewText'] = dask_df['reviewText'].apply(detect_gib, meta = ('reviewText', 'string'))
dask_df = dask_df.dropna()

#spellcheck

check = Speller(lang = 'en')

def fixSentence(sentence):
    return check(sentence)

dask_df['reviewText'] = dask_df['reviewText'].apply(fixSentence, meta = ('reviewText', 'string'))

dask_df = dask_df.compute()

dd.to_csv(dask_df, 'spellchecked.csv', single_file = True)
