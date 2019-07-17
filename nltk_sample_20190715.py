# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:03:10 2019

@author: USER
"""

 

# -*- coding: utf-8 -*-

"""

Created on Mon Jul 15 13:53:24 2019

https://medium.com/pyladies-taiwan/nltk-%E5%88%9D%E5%AD%B8%E6%8C%87%E5%8D%97-%E4%B8%80-%E7%B0%A1%E5%96%AE%E6%98%93%E4%B8%8A%E6%89%8B%E7%9A%84%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E5%B7%A5%E5%85%B7%E7%AE%B1-%E6%8E%A2%E7%B4%A2%E7%AF%87-2010fd7c7540

@author:  

"""

 

import nltk

#nltk.download()

from nltk.book import *

 
#! pip install jieba
import jieba.analyse

import matplotlib.pyplot as plt

 

from matplotlib.font_manager import _rebuild

_rebuild()

 

'''

NLTK 的畫圖是引用自 matplotlib 的套件，左邊縱軸的中文會是亂碼，有以下兩種解法：

解法一

每次都執行 plt.rcParams['font.sans-serif'] = ‘SimHei’ 的參數設定，但前提是需下載 SimHei 這個字體，並將 SimHei.ttf 放到你的 matplotlib 資料夾，讓畫圖時可以引用。

 

解法二

在 matplotlibrc 設定參數，之後不需要每次執行額外的程式，

下載 SimHei.ttf 放到你的 matplotlib 資料夾

到文件 matplotlibrc (在 matplotlib/mpl-data/fonts 目錄下面可以找到)，裡面修改下面三項配置：

> font.family : sans-serif

>font.sans-serif : SimHei, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif

>axes.unicode_minus : False # 解決負號(-) 顯示為方塊的問題

'''

 

 

'''

*** Introductory Examples for the NLTK Book ***

Loading text1, ..., text9 and sent1, ..., sent9

Type the name of the text or sentence to view it.

Type: 'texts()' or 'sents()' to list the materials.

text1: Moby Dick by Herman Melville 1851

text2: Sense and Sensibility by Jane Austen 1811

text3: The Book of Genesis

text4: Inaugural Address Corpus

text5: Chat Corpus

text6: Monty Python and the Holy Grail

text7: Wall Street Journal

text8: Personals Corpus

text9: The Man Who Was Thursday by G . K . Chesterton 1908

'''

 

#====================搜尋字詞：顯現字詞出現的上下文

#book.concordance()

text3.concordance("lived")

#Displaying 25 of 25 matches:

 

#=============================================找近似字

#book.similar()、book.common_contexts()

text1.similar("monstrous")  #怪異

'''

true contemptible christian abundant few part mean careful puzzled

mystifying passing curious loving wise doleful gamesome singular

delightfully perilous fearless

'''

 

# 回頭檢視結構

text1.common_contexts(["monstrous","abundant"])

#most_and

 

#==================================lexical_diversity 相異字詞長度/總字詞長度

 

# 相異字詞

set(text4)

# 相異字詞排序

sorted(set(text4))

# 定義詞彙多樣性的函數

def lexical_diversity(text):

    print('分子len(set(text))=', len(set(text)))

    print('分母len(text)=', len(text))

    return len(set(text)) / len(text)

 

lexical_diversity(text4)

# Result 0.06617622515804722

 

 

#===============================dispersion_plot詞彙分布圖

#book.dispersion_plot()

 

# 構造文本的詞彙分佈圖

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America", "liberty", "constitution"])

 

# sent1 與 sent2 內容

sent1

# ['Call', 'me', 'Ishmael', '.']

sent2

# ['The', 'family', 'of', 'Dashwood', 'had', 'long', 'been', 'settled', 'in', 'Sussex', '.']

# 文本的結合

sent1 + sent2

 

#['Call', 'me', 'Ishmael', '.', 'The', 'family', 'of', 'Dashwood', 'had', 'long', 'been', 'settled', 'in', 'Sussex', '.']

 

 

 

# 搜尋字詞

#lyrics 是book的概念

# 開檔,loop 每一row

#UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d in position 5: character maps to <undefined>
with open("data/mayday.txt", encoding="utf8") as f1:
    for line in f1:
        lyrics= nltk.text.Text(jieba.analyse.extract_tags(line))
        lyrics.concordance("我們")

 

# 近似字

# 開檔,一口氣讀完

raw = open("data/mayday.txt", encoding="utf8").read()

#jieba.cut 是做中文斷詞， nltk.text.Text 讓文本成為 NLTK 可以吃的格式

lyrics = nltk.text.Text(jieba.cut(raw))

lyrics.similar("我們")  #近似字

lyrics.common_contexts(["我們","我"])  #回頭檢視結構

 

 

#詞彙多樣性

#使用的 single.txt 為單一歌詞文本 (五月天的戀愛 ing )，透過前面介紹的函式，每一首歌詞都可以去計算詞彙豐富度

#len(set(single)) 為戀愛 ing 這首歌詞的相異字詞長度( 76)， len(single) 為總字詞長度( 244) ，兩者相除得到詞彙多樣性的值為 0.31 。

raw = open("data/single.txt", encoding="utf8").read()

single = nltk.text.Text(jieba.cut(raw))

print(type(single))

lexical_diversity(single)

 

#詞彙分布圖, 使用 single.txt單一歌詞文本

#將「戀愛」、「ing」、「happy」、「love」這幾個副歌字詞用 single.dispersion_plot 做呈現，

#看關鍵歌詞之間的先後順序，以及頻率分佈

 

# 呈現中文

 

plt.figure(figsize=(10, 5))

plt.rcParams['font.sans-serif'] = 'SimHei'

# 詞彙分佈圖

single.dispersion_plot(["love","戀愛","ing","happy"])


