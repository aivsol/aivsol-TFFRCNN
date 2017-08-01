import sys
f = open('all.txt')

data = {}
for l in f:
    t = l.split(";")[0]
    if t in data:
        data[t] += [l]
    else:
        data[t] = [l]
f.close()
train = dict(data.items()[0:380])
test = dict(data.items()[380:506])

'''
print ('\n'*10)
for k in sorted(train):
    print k, ':', train[k]

print ('\n'*10)
for k in sorted(test):
    print k, ':', test[k]
'''

# Sanity Check
for k in test:
    if k in train:
        print 'Not a valid train/test split'
        sys.exit()
print 'Success'
f = open("train.txt", "w")
for k in sorted(train):
    for l in train[k]:
        f.write(l)
f.close()
f = open('test.txt', "w")
for k in sorted(test):
    f.write(k.split(".")[0] + "\n")
f.close()
