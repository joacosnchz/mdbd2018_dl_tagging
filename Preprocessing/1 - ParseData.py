# IMPORTS
import xml.etree.ElementTree as ET
from pathlib import Path
import os
import msgpack

'''
parseData takes the following arguments:

    Origin: the path of the folder where the input data is contained. The folder must
            be arranged with one subfolder per forum, each containing its Posts.xml file


    Destination: Generally the same as Origin, it must be a folder with the same subfolder
                 layout. It will save three files per subfolder:

                        - WordDict: the word corpus for that forum

                        - WordCount: the data on how many times each word in wordDict
                                     appears in that forum

                        - Posts: the actual result, an array which contains every post
                                 in the forum as an array of words.


    Size: An integer indicating how many posts should be taken of each forum. If any forum
          has less posts than this number, an IndexOutOfRange exception will be raised
'''



# GLOBALS
wordDict = []
wordCount = []


def parseData(Origin, Destination, size):
    global wordDict
    global wordCount
    # LOADING DATA

    print("Loading data...")
    # os.walk returns an array of 3-tuples where the 0th item of the 3-tuples is the name of the folder preceeded by ".//"
    folders = [f[2:] for f in[i[0] for i in os.walk(Origin)][1:]]

    for f in range(len(folders)):
        print(''.join(("Parsing folder "+str(f)+" out of ",str(len(folders))+": ",folders[f])))
        r = []
        # getroot() gets the data from the xml file, and is structured as an array of posts
        postsRoot = ET.parse(Path(folders[f]) / "Posts.xml" ).getroot()
        index = 0

        # PARSING
        for i in range(size):
            # the attribute "Body" contains the body of the post, in html format. It needs to be parsed by parseBody()
            r.append(parseBody(postsRoot[i].attrib["Body"]))
            index += 1
            print(''.join((str(index),"/",str(size))))

    # SAVING RESULTS
        print("Saving Results")
        with open(Path(Destination)/folders[f]/("WordDict.msgpack"), 'wb+') as outfile:
            msgpack.pack(wordDict, outfile)
        with open(Path(Destination)/folders[f]/("WordCount.msgpack"), 'wb+') as outfile:
            msgpack.pack(wordCount, outfile)
        with open(Path(Destination)/folders[f]/"Posts.msgpack", 'wb+') as outfile:
            msgpack.pack(r, outfile)
        del r
        del postsRoot
        del index
        wordCount = []
        wordDict = []

    #multiHot(Origin, Destination)

def parseBody(body):
    # this function takes a string and eliminates any html tag or punctuation mark in it
    # then it stores each word as an item in an array and returns it
    # it also stores any word found on wordDict, as well as its respective number of
    # ocurrences on wordCount
    global wordDict
    global wordCount
    b = True
    r = ''
    for c in range(len(body)):
        if body[c] == "<":
            b = False
        if b:
            r += body[c]
        if body[c] == ">":
            b = True
    newBody =  ''.join([r[i] for i in range(len(r)) if(r[i] not in '.,-()[]-_;:\'\"\\/<>{{}}*+=!"£$%^&*<>?’# ')]).replace('\n','').replace("  "," ").lower()
    result = []
    i = 0
    for c in range(len(newBody)):
        if newBody[i] == ' ':
            a = newBody[0:i].replace(" ","")
            if a not in wordDict:
                wordDict.append(a)
                wordCount.append(1)
            else:
                wordCount[wordDict.index(a)] += 1
            result.append(a)
            newBody = newBody[i:]
            i = 0
        i+=1
    a = newBody.replace(" ","")
    if a not in wordDict:
        wordDict.append(a)
        wordCount.append(1)
    else:
        wordCount[wordDict.index(a)] += 1
    result.append(a)
    return result

parseData(Path("."),Path("."),5000)
