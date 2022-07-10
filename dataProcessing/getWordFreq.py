import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def count_words(s):
    # 词频统计
    lst = re.split('\<+|\>+|\|+|\'+|\d|\`+|\#+|\s+|\,+|\.+|\!+|\:+|\?+|\;+|\(+|\)+|\-+|\_+|\=+|\++|\“+|\、+|\/+|\{+|\}+|\”+|\：+|\。+|\“+|\[+|\]+|\【+|\】+|\—+|\%+|\"+',s)
    diccount = {}
    cnt = 0
    for i in lst:
        if i == ' ' or i =='' or len(i)==1:
            continue
        cnt += 1
        if i not in diccount:
            diccount[i] = 1  # 第一遍字典为空 赋值相当于 i=1，i为words里的单词
            # print(diccount)
        else:
            diccount[i] += 1
#     print(diccount)
#     print(cnt)
    for key in diccount:
        diccount[key] = round(100*diccount[key]/cnt,2)
    return diccount

def plot_word_cloud(s):
    # 画词云图
    wordcloud = WordCloud(background_color="white", \
                          width=400, \
                          height=300, \
                          max_font_size=80, \
                          contour_width=3, \
                          contour_color='steelblue'
                          ).generate(s)
    plt.figure(figsize=(13, 7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def get_word_freq(s):
    s1 = s.lower()
    # print(s1)
    dic1 = count_words(s1)
    dic1 = sorted(dic1.items(), key=lambda d: d[1], reverse=True)
    print(dic1[0:100])
    plot_word_cloud(s1)
