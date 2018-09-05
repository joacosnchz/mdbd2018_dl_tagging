from pathlib import Path
import os
import msgpack
from collections import defaultdict

'''
merge takes the following arguments:

    Origin: the path of the folder where the input data is contained. The folder must
            be arranged with one subfolder per forum, each containing its WordDict and
            WordCount files

    Destination: Generally the same as Origin. It's the path where the result will be
                 saved. Merge will take the WordDics and WordCounts of every subfolder
                 and merge them into one saved in Destination. 
'''


def merge(Origin, Destination):
    folders = [f[2:] for f in[i[0] for i in os.walk(Origin)][1:]]
    newWordDict = []
    newWordCount = []
    for f in range(len(folders)):
        print(''.join(("Merging folder "+str(f)+" out of ",str(len(folders))+": ",folders[f])))
        with open(Path(Origin)/folders[f]/"WordDict.msgpack","rb") as worddict, open(Path(Origin)/folders[f]/"WordCount.msgpack","rb") as wordcount:
            wd = msgpack.unpack(worddict)
            wc = msgpack.unpack(wordcount)
            for i in range(len(wd)):
                if wd[i] in newWordDict:
                    index = newWordDict.index(wd[i])
                    newWordCount[index] += wc[i]
                else:
                    newWordDict.append(wd[i])
                    newWordCount.append(wc[i])
            del wd
            del wc
    with open(Path(Destination)/"wordCount.msgpack", 'wb+') as outfile:
        msgpack.pack(newWordCount, outfile)
    with open(Path(Destination)/"wordDict.msgpack", 'wb+') as outfile:
        msgpack.pack(newWordDict, outfile)

merge('.','.')
