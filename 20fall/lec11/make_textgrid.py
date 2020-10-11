import numpy as  np
import praatio.tgio

entryList = []
with open('data/LDC93S1.phn') as f:
    for line in f:
        w = line.strip().split()
        s = float(w[0])/16000
        e = float(w[1])/16000
        entryList.append(praatio.tgio.Interval(s,e,w[2].upper()))

wordList = []
with open('data/LDC93S1.wrd') as f:
    for line in f:
        w = line.strip().split()
        s = float(w[0])/16000
        e = float(w[1])/16000
        wordList.append(praatio.tgio.Interval(s,e,w[2].upper()))

phn = praatio.tgio.IntervalTier(name='phn',entryList=entryList)
wrd = praatio.tgio.IntervalTier(name='wrd',entryList=wordList)
tg = praatio.tgio.Textgrid()
tg.addTier(phn)
tg.addTier(wrd)
tg.save('data/LDC93S1.TextGrid')

        
