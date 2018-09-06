import glob
import os

for file in glob.glob('./*.7z'):
    folder= file[2:][:-3]
    os.system( ''.join(('mkdir ',folder)))
    os.system( ''.join(('7z x ',file,' -o',folder)))
    for filename in os.listdir(folder):
        if filename != 'Posts.xml':
            os.system( ''.join(('del ', folder,'\\',filename)))
