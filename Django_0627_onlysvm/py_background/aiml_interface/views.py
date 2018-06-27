import aiml
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import time

from django.views.decorators.csrf import csrf_exempt

encoding = 'utf-8'
file_dir=os.getcwd()+'/aiml_interface/alice'

os.chdir(file_dir)  # 切换工作目录到alice文件夹下，视具体情况而定
varconf = file_dir + '/' + 'varconf.txt'
varconf_pre = file_dir + '/' + 'varconf_pre.txt'
quesconf = file_dir + '/' + 'questions.txt'
alice = aiml.Kernel()
alice.learn("startup.xml")
alice.respond('LOAD ALICE')

####Return chat message####
def chat_proc(mess_in, sessionId):
    mess_out = alice.respond(mess_in, sessionId)
    return mess_out

####var replace####
def var_proc(conf, str_in):
    fopen = open(conf, 'r')  # r 代表read
    for eachLine in fopen:
        varlist = eachLine.strip('\n').split('#|#')
        if varlist[0] in str_in:
            str_in = str_in.replace(varlist[0], varlist[1])
    fopen.close()
    return str_in

####Similarity judge####
def ques_proc(str_in, threshold_d, threshold_u):
    fopen = open(quesconf, 'r')  # r 代表read
    queslist = []
    for eachLine in fopen:
        queslist.append(eachLine.strip('\n').strip('?').strip('.').strip('!').strip('？').strip('。').strip('！'))
    fopen.close()
    sentout = {}
    for question in queslist:
        vocab = {}
        for word in (str_in.strip('?').strip('.') + ' ' + question).split():
            vocab[word] = 0
        vectorizer = CountVectorizer(vocabulary=vocab.keys())  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        aa = vectorizer.fit_transform([str_in])
        bb = vectorizer.fit_transform([question])
        sentout[question] = cosine_similarity(aa, bb)[0][0]
    selected = max(sentout.items(), key=operator.itemgetter(1))[0]

    if sentout[selected] >= threshold_d and sentout[selected] < threshold_u:
        return [1, selected]
    else:
        return [0, str_in]

####Clear history message####
def mess_old_clear(mess_dic, sec_gap):
    for mess_key in mess_dic.keys():
        curr_time = time.time()
        if curr_time - mess_dic[mess_key][1] >= sec_gap:
            del mess_dic[mess_key]
mess_old={}
def test(sessionId, mess):
    try:
        ####Question judge: Yes####
        if mess == 'Yes' or mess == 'yes' or mess == 'YES' or mess == 'YEs' or mess == 'YeS' or mess == 'yES' or mess == 'YeS' or mess == 'yEs':
            print('recieve yes')
            mess_send = chat_proc(mess_old[sessionId][0], sessionId)
        else:
            ####question process####
            mess_proc = ques_proc(mess, 0.8, 0.99)

            ####Answer process####
            if mess_proc[0] == 1:
                mess_send = 'If you want to ask questions like this: "' + mess_proc[1] + '"? Please type "Yes".'
                mess_old[sessionId] = [mess_proc[1], time.time()]
                mess_old_clear(mess_old, 60)
            else:
                ####Process: Have been\ has been####
                mess_proc[1] = var_proc(varconf_pre, mess_proc[1])
                mess_send = chat_proc(mess_proc[1], sessionId)

        ####Var in answer replace####
        if re.search('\${', mess_send) is not None:
            mess_send = var_proc(varconf, mess_send)
    except:
        mess_send = "I don't understand you! Please ask me another question."
    return mess_send


from django.shortcuts import render
from django.http import HttpResponse

from .forms import AddForm
import os

@csrf_exempt
def java_1(request):
    form = AddForm(request.POST)  # form 包含提交的数据

    if form.is_valid():  # 如果提交的数据合法
        sentence = form.cleaned_data['sentence'].strip()
        sessionid = form.cleaned_data['sessionid'].strip()

        ans=test(sessionid, sentence)

        ss = {'ans':ans}
        return HttpResponse(str(ss))

@csrf_exempt
# 接口1
def java(request):
    if request.method == 'POST':  # 当提交表单时

        form = AddForm(request.POST)  # form 包含提交的数据

        if form.is_valid():  # 如果提交的数据合法
            sentence = form.cleaned_data['sentence'].strip()
            sessionid = form.cleaned_data['sessionid'].strip()

            ans = test(sessionid, sentence)

            return render(request, 'aiml_test.html',
                          {'form': form, 'classifier': ans})

    else:  # 当正常访问时
        form = AddForm()
    return render(request, 'aiml_test.html', {'form': form})