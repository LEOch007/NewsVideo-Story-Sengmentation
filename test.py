import json
import meta
import exfeature
import time
import pandas as pd

# constant parameters
inpath = './data/input/'
outpath = './data/output/'


# read json data -> sentences
def Jsonhandler(inname):
    with open(inname,'r') as f:  #read json file
        data = json.load(f)
    # sentence instances
    sentence_list = []
    for i in range(len(data)):
        ibg = int(data[i]['bg'])//1000 # ms->s
        ied = int(data[i]['ed'])//1000 # ms->s
        sonebest = data[i]['onebest']

        if (ibg==0) or (ied==0): continue
        waste_list = ['嗯','啊','吖','呀','噢','哇','唉','哎','哦']
        if sonebest in waste_list: continue

        ss = meta.Sentences()
        ss.setbg(ibg)
        ss.seted(ied)
        ss.setonebest(sonebest)
        sentence_list.append(ss)
    return sentence_list


# get file name
def get_name(t):
    return 'hdcctv1_' + t[0].split('/')[-1]


# write csv file
def write_csv(sentences,outname):
    with open(outname,'w') as f:
        for i in range(len(sentences)):
            ss = str(sentences[i].getbg()) +','+ str(sentences[i].getkey()) +','+ str(sentences[i].gettinterval()) +','+ str(sentences[i].getsimscore()) +','+ str(sentences[i].getdeepscore()) + '\n'
            f.write(ss)


# main function
if __name__ == '__main__':
    gt_file_path_list = pd.read_csv('./data/gt_file_path', header=None)
    file_name_list = gt_file_path_list.apply(get_name,axis=1)

    # go through each file
    for filename in file_name_list:

        # ---- main procedure ---- #
        time_start = time.time()

        inname = inpath + filename
        outname = outpath + filename

        sentences = Jsonhandler(inname) # read json data
        code = -1  # state code
        fextract = exfeature.Feature_extract()

        # feature: keyword
        for i in range(len(sentences)):
            sentences[i].keyword()

        # feature: time span
        code = fextract.timedetect(sentences)
        assert code==0

        # feature: similarity score
        code = fextract.compute_simscore(sentences, 6)
        assert code==0

        # feature: deep score
        code = fextract.compute_deepscore(sentences)
        assert code==0

        time_end = time.time()
        print('time cost: ', time_end-time_start, 's')

        # output
        write_csv(sentences, outname)