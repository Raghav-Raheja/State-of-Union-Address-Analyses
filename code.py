# coding: utf-8
import os
import nltk
#from nltk.corpus import PlainTextCorpusReader
 nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
from nltk.collocations import *


os.getcwd()
stateunion1 = PlaintextCorpusReader('.','.*\.txt')
stateunion1_load = stateunion1.raw('Rachael.txt')
stateunion1_load = open('Rachael.txt')
rawtext = stateunion1_load.read()

stateunion1_tokenize = nltk.word_tokenize(rawtext)
#stateunion_lower[:120]
#stateunion_tokenize[:20]
stateunion1_tokenize[:20]
stateunion1_lower = [w.lower() for w in stateunion1_tokenize]
stateunion1_lower[:50]
stateunion1_chara = [w for w in stateunion1_lower if w.isalpha()]
stateunion1_chara[:50]
stopwords = nltk.corpus.stopwords.words('english')
stateunion1_stoppedwords = [w for w in stateunion1_chara if w not in stopwords]
stateunion1_stoppedwords[:50]
stateunion_fredist = FreqDist(stateunion1_stoppedwords)
for val in stateunion_fredist.keys():
    stateunion_fredist[val] = stateunion_fredist[val]/len(stateunion1_stoppedwords)

stateunion1_keys  = list(stateunion_fredist.keys())
stateunion1_keys[:20]
       
topkeys_stateunion1 = stateunion_fredist.most_common(50)
for pair1 in topkeys_stateunion1:
    print(pair1)
    
bigram_m_stateunion1 = nltk.collocations.BigramAssocMeasures()
stateunion1_finder = BigramCollocationFinder.from_words(stateunion1_stoppedwords)
stateunion1_scored = stateunion1_finder.score_ngrams(bigram_m_stateunion1.raw_freq)
for bscore1 in stateunion1_scored[:50]:
    print(bscore1)
    
stateunion1_finder.apply_freq_filter(5)
stateunion1_scoredp = stateunion1_finder.score_ngrams(bigram_m_stateunion1.pmi)
for bscore2 in stateunion1_scoredp[:50]:
    print(bscore2)
    
stateunion2_load = PlaintextCorpusReader('.','.*\.txt') 
stateunion2 = stateunion2_load.raw('state_union_part2.txt')
stateunion2_tokenize = nltk.word_tokenize(stateunion2)
len(stateunion2_tokenize)
stateunion2_tokenize[:20]
stateunion2_lower = [w.lower() for w in stateunion2_tokenize]
stateunion2_chara = [w for w in stateunion2_lower if w.isalpha()]
stateunion2_chara[:50]
stateunion2_stopped = [w for w in stateunion2_chara if w not in stopwords]
stateunion2_stopped[:50]
stateunion2_freqdist = FreqDist(stateunion2_stopped)

for val in stateunion2_freqdist.keys():
    stateunion2_freqdist[val] = stateunion2_freqdist[val]/len(stateunion2_stopped)

stateunion2_keys = list(stateunion2_freqdist.keys())
stateunion2_keys[:10]
stateunion2_topkeys = stateunion2_freqdist.most_common(50)
for bscore3 in stateunion2_topkeys:
    print(bscore3)
    
stateunion2_b_measure = nltk.collocations.BigramAssocMeasures()
stateunion2_finder = BigramCollocationFinder.from_words(stateunion2_stopped)
stateunion2_scored = stateunion2_finder.score_ngrams(stateunion2_b_measure.raw_freq)
for score1 in stateunion2_scored[:50]:
    print(score1)
    
stateunion2_finder.apply_freq_filter(5)
stateunion2_scoredp = stateunion2_finder.score_ngrams(stateunion2_b_measure.pmi)
for score2 in stateunion2_scoredp[:50]:
    print(score2)
    

#%save -r mysession 1-99999
