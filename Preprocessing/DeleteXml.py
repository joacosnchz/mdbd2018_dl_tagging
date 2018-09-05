import os
folders = [f[2:] for f in[i[0] for i in os.walk('.')][1:]]
for f in folders:
    os.system( ''.join(('del ', f,'\\Posts.xml')))
