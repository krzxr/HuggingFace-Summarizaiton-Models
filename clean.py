import os 

path = 'science_daily/'
outP = 'sd/'
punctuations = ['.', '!', '?']
countF = 0
for filename in os.listdir(path):
    print(filename)
    if filename[-4:] == '.txt' and "abstract" not in filename:
        print('editing')
        # get files 
        absFn = filename[:-4] + "_abstract.txt"
        f_txt = open(path+filename, 'r', encoding='utf-8')
        f_abs = open(path+absFn, 'r', encoding='utf-8')
        abstract = f_abs.read().strip().lower()
        article = f_txt.read().strip().lower()
        f_txt.close()
        f_abs.close()
        # clean 
        for p in punctuations:
            abs_sentences = abstract.split(p)
            for s in abs_sentences:
                if len(s) > 10:
                    article = article.replace(s, '', 1)
        # save files to new dir
        o_txt = open(outP+'txt/'+filename, 'wb')
        o_abs = open(outP+'abs/'+absFn, 'wb')
        o_txt.write(article.encode('ascii', 'ignore'))
        o_abs.write(abstract.encode('ascii', 'ignore'))
        o_txt.close()
        o_abs.close()
        countF+=1
print(countF)