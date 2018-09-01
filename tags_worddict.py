import msgpack

def get_words(a):
	r = []
	aux = ''
	for c in a:
		if c == ' ':
			r.append(aux.encode('ascii'))
			aux = ''
		else:
			aux+= c
	r.append(aux.encode('ascii'))
	return r

def get_multihot(a):
	r = get_words(a)
	wD = msgpack.unpack(open("./input/WordDict.msgpack","rb"))
	
	r = [1 if(wD[j] in r) else 0 for j in range(len(wD))]
	return r


