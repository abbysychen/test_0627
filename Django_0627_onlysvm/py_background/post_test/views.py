from django.shortcuts import render
from django.http import HttpResponse

from .forms import AddForm,AddForm2
import os

import pandas,nltk,pickle,pycrfsuite,re
import numpy as np
import re

from django.views.decorators.csrf import csrf_exempt
import psutil

path=os.getcwd()+'/post_test/'

# 记录输入的聊天记录，并存放在sentence.txt中
def write_sentence(path,sentence):
    with open(path+'sentence.txt', 'a') as f:
        f.write(sentence+'\n')

# 模型1：判断句子是否业务相关
f = open(path+'classifier_model.pickle', 'rb')    #打开训练好的SVM模型文件
classifier_model = pickle.load(f)      #载入打开的模型
f.close()  #关闭文件

outside_lexicon = pandas.read_excel(path+'0621_lexicon_by_4types.xlsx')   #将每个意图的特定词汇汇总成词典，作为其中一种特征
trend_show_lexicon = outside_lexicon[outside_lexicon.word_type == 'trend_show']
sales_show_lexicon = outside_lexicon[outside_lexicon.word_type == 'sales_search']
time_search_lexicon= outside_lexicon[outside_lexicon.word_type == 'time_search']
comments_search_lexicon= outside_lexicon[outside_lexicon.word_type == 'comments_search']

with open(path+'modle_1_feature.txt', 'r') as f:
    all_line_txt = f.readlines()
modle_1_fea=all_line_txt[0].split('|||')

def get_feature_from_sentence(sentence):
    word_list = sentence.split()
    word_list = set(word_list)

    # 初始化
    features = {}
    trend_show_cnt = 0
    sales_show_cnt = 0
    time_search_cnt = 0
    comments_search_cnt = 0

    # 统计一个句子是否含有raw_data中的词库中的词
    for word in modle_1_fea:
        features['contains({})'.format(word)] = (word in word_list)

    # 统计一个句子中，各种特属类别的单词有几个
    for word in word_list:
        if word in list(trend_show_lexicon['word']):  # 判断单词是否在trend_show中
            trend_show_cnt += 1
        if word in list(sales_show_lexicon['word']):  # 判断单词是否在sales_show中
            sales_show_cnt += 1
        if word in list(time_search_lexicon['word']):  # 判断单词是否在time_search中
            time_search_cnt += 1
        if word in list(comments_search_lexicon['word']):  #判断单词是否在comments_search中
            comments_search_cnt += 1

    # 储存统计结果
    features["trend_show_cnt"] = trend_show_cnt  # 将最终的trend_show_cnt 值作为特征
    features["sales_show_cnt"] = sales_show_cnt  # 将最终的sales_show_cnt 值作为特征
    features["time_search_cnt"] = time_search_cnt  # 将最终的time_search_cnt 值作为特征
    features["comments_search_cnt"] = comments_search_cnt  # 将最终的comments_search_cnt 值作为特征

    return features

'''
# 模型2：对输入句子进行意图识别
#模型2 - 第一部分：读取训练好的模型
f = open(path+'NaiveBayesClassifier_model.pickle', 'rb')  #朴素贝叶斯模型
NaiveBayesClassifier_model = pickle.load(f)
f.close()

f = open(path+'MaxentClassifier_model.pickle', 'rb')  #最大熵模型
MaxentClassifier_model = pickle.load(f)
f.close()

f = open(path+'xgb_model.pickle', 'rb')  #xgboost模型
xgb_model = pickle.load(f)
f.close()

#模型2 - 第二部分：根据输入句子生成指定格式的特征

#根据valuable_word生成特征
valuable_word= ['what',
 'is',
 'the',
 'overall',
 'sales',
 'volume',
 'in',
 'may',
 'how',
 'about',
 '2018',
 'top',
 '3',
 '10',
 'this',
 'year',
 'highest',
 'last',
 'month',
 'of',
 'auguest',
 '5series',
 '6series',
 '201801',
 'x5',
 'apr',
 'x',
 '1',
 '2015',
 'check',
 'wholesale',
 'show',
 'me',
 'our',
 'car',
 '5',
 'series',
 'tell',
 '2018.03',
 'from',
 'to',
 'august',
 'recently',
 'december',
 'let',
 'see',
 'amount',
 'by',
 'type',
 'area',
 'brands',
 'segment',
 'gkl+sedan',
 'digit',
 'retail',
 'much',
 '2017',
 'complete',
 'rate',
 'which',
 'region',
 'completed',
 'target',
 'that',
 'accomplished',
 'mission',
 'achieved',
 'goal',
 'model',
 'best-seller',
 'best',
 'seller',
 'best-selling',
 'selling',
 'sell',
 'has',
 'largest',
 'market',
 'share',
 'models',
 'with',
 'decreasing',
 'dropping',
 'got',
 'sold',
 'most',
 'trend',
 'bmw',
 'and',
 'its',
 'competitors',
 'oppenent',
 'competition',
 'position',
 'brand',
 'image',
 'contribution',
 'graph',
 'charts',
 'shows',
 'recent',
 'years',
 'performance',
 'competitor',
 'comparison',
 'compare',
 'different',
 'audi',
 'segmentation',
 'sale',
 'channel',
 'segement',
 'comments',
 'new',
 'x3',
 'internet',
 'x4',
 'x6',
 'there',
 'any',
 'interior',
 'exterior',
 'voice',
 'social',
 'listening',
 'source',
 'date',
 'do',
 'they',
 'like',
 'dislike',
 'consumer',
 'customer',
 'not']


#生成输入句子的特征
def get_feature(sentence):
    word_list=sentence.split()
    word_list=set(word_list)
    features={}
    for word in valuable_word:
        features['contains({})'.format(word)]=(word in word_list)
    return features

#模型3 - CRF命名实体判别
tagger = pycrfsuite.Tagger()
tagger.open(path+'CRF_model.crfsuite')

fea =  ['1',
 '10',
 '2015',
 '2017',
 '2018',
 '201801',
 '201803',
 '5',
 '5series',
 '6series',
 'Audi',
 'Complete',
 'Contribution',
 'DIGIT',
 'GKL+Sedan',
 'How',
 'Let',
 'May',
 'Show',
 'Shows',
 'Target',
 'Tell',
 'The',
 'What',
 'Which',
 'Wholesale',
 'X',
 'about',
 'accomplished',
 'achieved',
 'amount',
 'any',
 'apr',
 'area',
 'auguest',
 'august',
 'best-seller',
 'best-selling',
 'bmw',
 'by',
 'channel',
 'charts',
 'check',
 'comments',
 'compare',
 'comparison',
 'competition',
 'competitor',
 'competitors',
 'complete',
 'completed',
 'contribution',
 'date',
 'december',
 'decreasing',
 'different',
 'do',
 'dropping',
 'exterior',
 'from',
 'goal',
 'got',
 'graph',
 'has',
 'highest',
 'how',
 'image',
 'in',
 'interior',
 'internet',
 'is',
 'its',
 'largest',
 'last',
 'like',
 'listening',
 'may',
 'me',
 'mission',
 'most',
 'much',
 'new',
 'oppenent',
 'position',
 'rate',
 'recent',
 'recently',
 'sale',
 'see',
 'segement',
 'segment',
 'segmentation',
 'sell',
 'seller',
 'selling',
 'show',
 'social',
 'source',
 'tell',
 'that',
 'there',
 'they',
 'this',
 'trend',
 'type',
 'voice',
 'what',
 'which',
 'with',
 'x',
 'x3',
 'x6',
 'years']


def word2features(sent, i):
    """
    Input Parameters——sent: a string;
                           i: indicates the place of the string
    Output Parameters——the features of the ith word of the string

    """
    word = sent[i][0]  # 选择输入句子的第i个词
    postag = sent[i][1]  # 选择输入句子的第i个词的postag

    # 创造feature
    features = [
        'bias',  # bias?
        'word.lower=' + word.lower(),  # word的小写形式  如果word = 'Happy', 则本特征= 'happy'
        'word[-3:]=' + word[-3:],  # word的最后三个字母
        'word[-2:]=' + word[-2:],  # word的最后两个字母
        'word.isupper=%s' % word.isupper(),  # 是否整个单词都是大写的
        'word.istitle=%s' % word.istitle(),  # 单词的第一个字母是否是大写
        'word.isdigit=%s' % word.isdigit(),  # 检测单词是否只由数字组成
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features.append("count({})=".format(letter)+ str(word.lower().count(letter)))
        features.append("has({})=".format(letter)+ str(letter in word.lower()))

    for fea_word in fea:  # 判断本单词是否在特征词典中出现；在特征词典中出现的次数
        features.append("count({})=".format(fea_word) + str(word.lower().count(fea_word)))
        features.append("has({})=".format(fea_word) + str(fea_word in word.lower()))

    if i > 0:  # 如果不是第一个词，提取前一个词的特征
        word1 = sent[i - 1][0]  # 用word1记录本单词的前一个词
        postag1 = sent[i - 1][1]  # 用postag1记录本单词的前一个词的词性

        features.extend([
            '-1:word.lower=' + word1.lower(),  # 前一个词的小写形式
            '-1:word.istitle=%s' % word1.istitle(),  # 前一个词的首字母是否大写
            '-1:word.isupper=%s' % word1.isupper(),  # 前一个词是否都是大写
            '-1:postag=' + postag1,  # 前一个词的词性
            '-1:postag[:2]=' + postag1[:2],  # 前一个词的统一词性
        ])
    else:
        features.append('BOS')

    if i > 1:  # 如果不是前两个词，提取第前2个的词的特征
        word2 = sent[i - 2][0]  # 用word2记录本单词的第前2个词
        postag2 = sent[i - 2][1]  # 用postag2记录本单词的前2个词的词性

        features.extend([
            '-2:word.lower=' + word2.lower(),  # 第前2个词的小写形式
            '-2:word.istitle=%s' % word2.istitle(),  # 第前2个词的首字母是否大写
            '-2:word.isupper=%s' % word2.isupper(),  # 第前2个词是否都是大写
            '-2:postag=' + postag2,  # 第前2个词的词性
            '-2:postag[:2]=' + postag2[:2],  # 第前2个词的统一词性
        ])

    if i < len(sent) - 1:  # 如果不是最后一个词
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    if i < len(sent) - 2:  # 如果单词在倒数第二个词之前
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]
        features.extend([
            '+2:word.lower=' + word2.lower(),
            '+2:word.istitle=%s' % word2.istitle(),
            '+2:word.isupper=%s' % word2.isupper(),
            '+2:postag=' + postag2,
            '+2:postag[:2]=' + postag2[:2],
        ])

    return features

def sent2tokens(sent):
    if len(sent[0])==3:
        return [token for token, postag, label in sent]
    else:
        return [token for token, label in sent]

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]



#正则匹配函数，从输入句子中判断并提取指定类型的值
f = open(path+'0608.match_lec.csv')
res = pandas.read_csv(f,header=None)
res.iloc[0,0]='model'
res.columns = ['slot_type','rule1','rule2','slot_value']

def zz_tz(sentence,slot_type):
    sentence=sentence.lower()
    returned_list = []
    res_slot_type = res[res['slot_type']==slot_type]
    for i in range(len(res_slot_type)):
        rule1 = res_slot_type.iloc[i][1]
        rule2 = res_slot_type.iloc[i][2]
        if len(re.findall((rule1),sentence))+len(re.findall(rule2,sentence))>0:
            returned_list.append( res_slot_type.iloc[i][3])
    return returned_list

def find_timepoint_from_input(sentence, slot_type='month'):
    sentence = sentence.lower()
    # 读取pattern
    f = open(path+'0612_lec_for_timepoint.csv')
    res = pandas.read_csv(f, header=None)
    res.columns = ['slot_type', 'rule', 'slot_value']
    month = []

    # 判断是否有月份的单词
    res_slot_type = res[res['slot_type'] == slot_type]
    for i in range(len(res_slot_type)):
        rule = res_slot_type.iloc[i][1]
        # print(rule)
        pattern = re.compile(rule)
        match = pattern.search(sentence, 0)
        if not match:
            continue
        else:
            s = match.start()
            e = match.end()
            month.append([res_slot_type.iloc[i][2], s, e])
    print("month info in the sentence ", month)

    # 提取句子中的所有数字
    year_value = 2018
    year_s = 0
    for index, match in enumerate(re.finditer(r'(\d+)', sentence)):
        s = match.start()
        print("s:", s)
        e = match.end()
        print("e:", e)
        value = sentence[s:e]
        print(value)

        if len(re.findall(r'(\d+)', sentence)) == 1 and len(sentence[s:e]) == 4:
            if int(sentence[s]) in [1, 2]:
                year_value = int(value)
                year_s = s
                print("there are year_value in the sentence")
        elif len(re.findall(r'(\d+)', sentence)) > 1:
            if len(sentence[s:e]) == 4:
                if int(sentence[s]) in [1, 2]:
                    year_value = int(value)
                    year_s = s
                    print("there are year_value in the sentence")
            else:
                month.append([value, s, e])
                print(month)

    # 如果year存在，计算提取的数字的位置的远近
    if year_s != 0 and len(month) > 1:
        print("have year info in the sentence")
        dist = []
        for i, info in enumerate(month):
            dist.append([abs(month[i][1] - year_s), month[i][0]])
        print(min(dist))
        month_value = int(min(dist)[1])

    elif year_s != 0 and len(month) == 1:
        print("1 year and 1 month")
        month_value = int(month[0][0])

    elif year_s != 0 and month == []:
        print("1 year and no month")
        month_value = None
    elif year_s == 0 and len(month) == 1:
        year_value = 2018
        month_value = int(month[0][0])
    else:
        year_value = None
        month_value = None
    if year_value == None:
        year_value = ''
    if month_value == None:
        month_value = ''

    return year_value, month_value
'''

# 网页
def index(request):
    if request.method == 'POST':  # 当提交表单时

        form = AddForm(request.POST)  # form 包含提交的数据

        if form.is_valid():  # 如果提交的数据合法
            sentence = form.cleaned_data['sentence'].strip()
            write_sentence(path,sentence)  #储存输入的句子

            punc = '[,>.?!\']'
            sentence = re.sub(punc, '', sentence)

            #sentence = sentence.replace('?','').replace('？','')
            classifier_model_s = classifier_model.classify(get_feature_from_sentence(sentence))
            
            if classifier_model_s == 0:
                classifier_model_s = 'F'
            else:
                classifier_model_s = 'True'

            model2_features=get_feature(sentence)

            NaiveBayes_s =NaiveBayesClassifier_model.classify(model2_features)
            MaxentClassifier_s=MaxentClassifier_model.classify(model2_features)
            xgboost_s=xgb_model.predict(list(model2_features.values()))[0]

            example_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
            sentence_s=' '.join(sent2tokens(example_sent))
            sentence_post_s=' '.join(tagger.tag(sent2features(example_sent)))

            if NaiveBayes_s == MaxentClassifier_s:
                tp = NaiveBayes_s
            elif NaiveBayes_s == xgboost_s:
                tp = NaiveBayes_s
            elif MaxentClassifier_s == xgboost_s:
                tp = MaxentClassifier_s
            else:
                tp = NaiveBayes_s

            return render(request, 'index.html', {'form': form,'tp':tp, 'classifier': classifier_model_s, 'na_string': NaiveBayes_s,'ma_string':MaxentClassifier_s,'xg_string':xgboost_s,'sen_string':sentence_s,'pos_string':sentence_post_s})

    else:  # 当正常访问时
        form = AddForm()
    return render(request, 'index.html', {'form': form})

#接口1
@csrf_exempt
def java(request):

    info =psutil.virtual_memory()
    print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
    print (u'总内存：',info.total)
    print (u'内存占比：',info.percent)
    print (u'cpu个数：',psutil.cpu_count())

    form = AddForm(request.POST)  # form 包含提交的数据

    if form.is_valid():  # 如果提交的数据合法
        sentence = form.cleaned_data['sentence'].strip()

        print(sentence)
        write_sentence(path, sentence)

        punc = '[,>.?!\']'
        sentence = re.sub(punc, '', sentence)

        # #0602正则
        # year_return,month_return=find_timepoint_from_input(sentence)  #从输入句子中提取时间
        #
        # competitor_list=zz_tz(sentence,'competitor')  #从输入句子中提取竞争者
        # if len(competitor_list)>0:
        #     competitor_re='T'
        # else:
        #     competitor_re='F'


        #SVM分类模型结果
        classifier_model_s=classifier_model.classify(get_feature_from_sentence(sentence))
        
        if classifier_model_s==0:
            classifier_model_s='F'
        else:
            classifier_model_s='T'     


        # #意图识别模型结果
        # model2_features = get_feature(sentence)
        #
        # NaiveBayes_s = NaiveBayesClassifier_model.classify(model2_features)
        # MaxentClassifier_s = MaxentClassifier_model.classify(model2_features)
        # print(np.array(list(model2_features.values())))
        #
        # xgboost_s = xgb_model.predict(list(model2_features.values()))[0]
        #
        # if NaiveBayes_s==MaxentClassifier_s:
        #     tp=NaiveBayes_s
        # elif NaiveBayes_s==xgboost_s:
        #     tp = NaiveBayes_s
        # elif MaxentClassifier_s==xgboost_s:
        #     tp = MaxentClassifier_s
        # else:
        #     tp=NaiveBayes_s
        #
        # #命名实体CRF判别结果
        # example_sent = nltk.pos_tag(nltk.word_tokenize(sentence))
        # sentence_s = ' '.join(sent2tokens(example_sent))
        # sentence_post_s = ' '.join(tagger.tag(sent2features(example_sent)))

        ssss = {'year_return': '2018', 'month_return': 'may', 'competitor': 'yes', 'classifier': classifier_model_s, 'tp': 'Sales_search','sentence':'1','sentence_post': '1'}
        print(ssss)
        return HttpResponse(str(ssss))

@csrf_exempt
def java_1(request):
    form = AddForm2(request.POST)  # form 包含提交的数据

    if form.is_valid():  # 如果提交的数据合法
        sentence = form.cleaned_data['sentence'].strip()
        slot_type = form.cleaned_data['slot_type'].strip()

        if slot_type=='model':
            ss=zz_tz(sentence,slot_type)
            ss={slot_type:ss}
            return HttpResponse(str(ss))

        elif slot_type=='Time':
            slot_type='month'
            year_return,month_return=find_timepoint_from_input(sentence)
            ss = {'year_return':str(year_return),'month_return':str(month_return)}
            return HttpResponse(str(ss))

        else:
            ss = {'slot_type':slot_type}
            return HttpResponse(str(ss))