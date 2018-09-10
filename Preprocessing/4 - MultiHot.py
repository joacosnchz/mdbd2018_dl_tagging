from pathlib import Path
import os
import msgpack

'''
multiHot takes the following arguments:

    Origin: the path of the folder where the input data is contained. The folder must
            contain a WordDict.msgpack file and also be arranged with one subfolder
            per forum, each containing its Posts.msgpack file,

    Destination: Generally the same as Origin, it must be a folder with the same subfolder
                 layout. It will save one OneHot.msgpack file per subfolder, which contains
                 an array of the posts of that forums in the form of a multihot array, a
                 0 representing it does not contain the corresponding word in wordDict and
                 a 1 representing it does.
'''


# GLOBALS
forumKeys = []

def multiHot(Origin, Destination):
    print("Generating multihot arrays")
    folders = [f[2:] for f in[i[0] for i in os.walk(Origin)][1:]]
    for f in range(len(folders)):
        print(''.join(("Creating MultiHot of folder "+str(f)+" out of ",str(len(folders))+": ",folders[f])))
        forumKeys.append([f,folders[f],[1 if (f == k) else 0 for k in range(len(folders))]])
        with open(Path(folders[f]) / "Posts.msgpack", "rb") as posts, open(Path(Origin)/"WordDict.msgpack","rb") as w:
            p,wD = msgpack.unpack(posts),msgpack.unpack(w)
            c,l = 0,len(p)
            r = []
            for i in range(l):
                print(''.join((str(c),'/',str(l))))
                r.append([[],[1 if (f == k) else 0 for k in range(len(folders))]])
                for j in range(len(wD)):
                    if wD[j] in p[i][0]:
                        r[i][0].append(j)
                c+=1
        with open(Path(Destination)/folders[f]/"OneHot.msgpack", 'wb+') as outfile:
            msgpack.pack(r, outfile)
        del p
        del wD
        del r
        del c
        del l
        

    with open(Path(Destination)/"forumKeys.msgpack", 'wb+') as outfile:
        msgpack.pack(forumKeys, outfile)

multiHot('.','.')
