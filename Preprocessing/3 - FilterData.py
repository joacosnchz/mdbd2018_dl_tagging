from pathlib import Path
import os
import msgpack

'''
filterData takes the following arguments:

    Origin: the path of the folder where the input data is contained. The folder must
            contain a WordCount and WordDict file, produced by merge.py

    Destination: Generally the same as Origin. If it is, it will override both
                 WordCount and WordDict files with the new filtered version.
                 If it isn't the new WordCount and WordDict will be saved in the
                 path specified.

    lowerThreshold: The minimun ammount of times a word must appear in the corpus
                    in order to not be filtered. Any word with a count below this
                    will be eliminated from the dictionary.

    upperThreshold: The maximum ammount of times a word must appear in the corpus
                    in order to not be filtered. Any word with a count above this
                    will be eliminated from the dictionary.
'''



def filterData(Origin,Destination,lowerThreshold,upperThreshold):
    print("Filtering results...")

    with open(Path(Origin)/"WordCount.msgpack", "rb") as wordCount, open(Path(Origin)/"WordDict.msgpack","rb") as wordDict:
        wC, wD = msgpack.unpack(wordCount),msgpack.unpack(wordDict)
        wordsToDelete = []
        length = len(wC)
        for i in range(length):
            print(str(i)+"/"+str(length))
            # 28 is the length of the longest word in the english language
            if (wC[i] < lowerThreshold or wC[i] > upperThreshold or len(wD[i]) > 28 or '\\' in str(wD[i])):
                wordsToDelete.append(i)

        print("Deleting filtered words")
        for i in wordsToDelete[::-1]:
            del wD[i]
            del wC[i]

        with open(Path(Destination)/"wordCount.msgpack", 'wb+') as outfile:
            msgpack.pack(wC, outfile)
        with open(Path(Destination)/"wordDict.msgpack", 'wb+') as outfile:
            msgpack.pack(wD, outfile)
        with open(Path(Destination)/"deletedWords.msgpack", 'wb+') as outfile:
            msgpack.pack(wordsToDelete, outfile)

filterData('.','.',3,36176) # nr of posts * 0.14470588
