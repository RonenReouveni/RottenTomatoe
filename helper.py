from textblob import TextBlob
from nltk.tokenize import word_tokenize


testLIST = ['you are neither funny nor smart', 'you are neither boring nor mean', 'it seems very good, but its actually just awful', 'it seems really bad, but I loved it', 'unlike how terrible its predacessor was, this was actually amazing']
myLabels =  ['negative', 'positive','negative','positive', 'positive']

tokenized_sents_test = [word_tokenize(i) for i in testLIST]
flippers = ['but', 'however', 'although']
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']





def customSentiment(tokenizedSentences):
    phraseS = []
    for sentence in tokenizedSentences:
        state = 0
        indexes = iter(range(len(sentence)))
        for i in indexes:
            if sentence[i] in flippers:
                state = 0
            state += (TextBlob(sentence[i]).sentiment[0])
            if sentence[i] in negationwords:
                try:
                    state -= (TextBlob(sentence[i+1]).sentiment[0])
                    next(indexes)
                except Exception:
                    pass
        phraseS.append(state)
    return phraseS

phraseSentiment_test = customSentiment(tokenized_sents_test)




def normalized(myList):
    normalSents = []
    for x in myList:
        normalSents.append((x-min(myList))/(max(myList)-min(myList)))
    return normalSents

#phraseSentiment_test = normalized(phraseSentiment_test)


boringSent = []
for sent in testLIST:
    boringSent.append(TextBlob(sent).sentiment[0])

print(myLabels)
print(phraseSentiment_test)
print(boringSent)
