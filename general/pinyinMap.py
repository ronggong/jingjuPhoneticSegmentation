#!/usr/bin/env python
# -*- coding: utf-8 -*-

initials    =[u'', u'ch', u'zh', u'r', u'c', u'b', u'd', u'g', u'f', u'h', u'k', u'j', u'm', u'l', u'n', u'q', u'p', u's', u'sh', u't', u'w', u'y', u'x', u'z']
finals      =[u'en', u'ei', u've', u'ai', u'uan', u'iu', u'ong', u'ao', u'an', u'uai', u'ang', u'iong', u'in', u'ia', u'ing', u'ie', u'er', u'iao', u'ian', u'eng', u'iang', u'ui', u'uang', u'a', u'e', u'i', u'o', u'uo', u'un', u'u', u'v', u'ue', u'ou', u'ua']

dic_pinyin_2_initial_final_map = {u'gu': {'initial': u'g', 'final': u'u'},
                                  u'guang': {'initial': u'g', 'final': u'uang'},
                                  u'qian': {'initial': u'q', 'final': u'ian'},
                                  u'fong': {'initial': u'f', 'final': u'ong'},
                                  u'ping': {'initial': u'p', 'final': u'ing'},
                                  u'zei': {'initial': u'z', 'final': u'ei'},
                                  u'zen': {'initial': u'z', 'final': u'en'},
                                  u'kong': {'initial': u'k', 'final': u'ong'},
                                  u'ge': {'initial': u'g', 'final': u'e'},
                                  u'chuang': {'initial': u'ch', 'final': u'uang'},
                                  u'tian ': {'initial': u't', 'final': u'ian'},
                                  u'lian': {'initial': u'l', 'final': u'ian'},
                                  u'liao': {'initial': u'l', 'final': u'iao'},
                                  u'rou': {'initial': u'r', 'final': u'ou'},
                                  u'shou': {'initial': u'sh', 'final': u'ou'},
                                  u'hai': {'initial': u'h', 'final': u'ai'},
                                  u'feng': {'initial': u'f', 'final': u'eng'},
                                  u'jie': {'initial': u'j', 'final': u'ie'},
                                  u'zong': {'initial': u'z', 'final': u'ong'},
                                  u'tong': {'initial': u't', 'final': u'ong'},
                                  u'han': {'initial': u'h', 'final': u'an'},
                                  u'hao': {'initial': u'h', 'final': u'ao'},
                                  u'jin': {'initial': u'j', 'final': u'in'},
                                  u'xiong': {'initial': u'x', 'final': u'iong'},
                                  u'lai': {'initial': u'l', 'final': u'ai'},
                                  u'tu': {'initial': u't', 'final': u'u'},
                                  u'seng': {'initial': u's', 'final': u'eng'},
                                  u'li': {'initial': u'l', 'final': u'i'},
                                  u'lv': {'initial': u'l', 'final': u'v'},
                                  u'mu': {'initial': u'm', 'final': u'u'},
                                  u'jiu': {'initial': u'j', 'final': u'iu'},
                                  u'ti': {'initial': u't', 'final': u'i'},
                                  u'cun': {'initial': u'c', 'final': u'un'},
                                  u'meng': {'initial': u'm', 'final': u'eng'},
                                  u'cui': {'initial': u'c', 'final': u'ui'},
                                  u'ta': {'initial': u't', 'final': u'a'},
                                  u'bin': {'initial': u'b', 'final': u'in'},
                                  u'pu': {'initial': u'p', 'final': u'u'},
                                  u'zhao': {'initial': u'zh', 'final': u'ao'},
                                  u'kui': {'initial': u'k', 'final': u'ui'},
                                  u'di': {'initial': u'd', 'final': u'i'},
                                  u'ya': {'initial': u'y', 'final': u'a'},
                                  u'dan': {'initial': u'd', 'final': u'an'},
                                  u'dao': {'initial': u'd', 'final': u'ao'},
                                  u'ye': {'initial': u'y', 'final': u'e'},
                                  u'dai': {'initial': u'd', 'final': u'ai'},
                                  u'da': {'initial': u'd', 'final': u'a'},
                                  u'bang': {'initial': u'b', 'final': u'ang'},
                                  u'fan': {'initial': u'f', 'final': u'an'},
                                  u'xiu': {'initial': u'x', 'final': u'iu'},
                                  u'ma': {'initial': u'm', 'final': u'a'},
                                  u'du': {'initial': u'd', 'final': u'u'},
                                  u'yu': {'initial': u'y', 'final': u'u'},
                                  u'yong': {'initial': u'y', 'final': u'ong'},
                                  u'gen': {'initial': u'g', 'final': u'en'},
                                  u'gua': {'initial': u'g', 'final': u'ua'},
                                  u'qu': {'initial': u'q', 'final': u'u'},
                                  u'shu': {'initial': u'sh', 'final': u'u'},
                                  u'zhen': {'initial': u'zh', 'final': u'en'},
                                  u'die': {'initial': u'd', 'final': u'ie'},
                                  u'gui': {'initial': u'g', 'final': u'ui'},
                                  u'guo': {'initial': u'g', 'final': u'uo'},
                                  u'shao': {'initial': u'sh', 'final': u'ao'},
                                  u'sang': {'initial': u's', 'final': u'ang'},
                                  u'zhu': {'initial': u'zh', 'final': u'u'},
                                  u'she': {'initial': u'sh', 'final': u'e'},
                                  u'ban': {'initial': u'b', 'final': u'an'},
                                  u'bao': {'initial': u'b', 'final': u'ao'},
                                  u'bai': {'initial': u'b', 'final': u'ai'},
                                  u'en': {'initial': u'', 'final': u'en'},
                                  u'zhuang': {'initial': u'zh', 'final': u'uang'},
                                  u'zi': {'initial': u'z', 'final': u'i'},
                                  u'xuan': {'initial': u'x', 'final': u'uan'},
                                  u'ken': {'initial': u'k', 'final': u'en'},
                                  u'shen': {'initial': u'sh', 'final': u'en'},
                                  u'ze': {'initial': u'z', 'final': u'e'},
                                  u'nuo': {'initial': u'n', 'final': u'uo'},
                                  u'lao': {'initial': u'l', 'final': u'ao'},
                                  u'nui': {'initial': u'n', 'final': u'ui'},
                                  u'chen': {'initial': u'ch', 'final': u'en'},
                                  u'sen': {'initial': u's', 'final': u'en'},
                                  u'er': {'initial': u'', 'final': u'er'},
                                  u'dian': {'initial': u'd', 'final': u'ian'},
                                  u'fou': {'initial': u'f', 'final': u'ou'},
                                  u'ying': {'initial': u'y', 'final': u'ing'},
                                  u'xing': {'initial': u'x', 'final': u'ing'},
                                  u'fang': {'initial': u'f', 'final': u'ang'},
                                  u'suo': {'initial': u's', 'final': u'uo'},
                                  u'sun': {'initial': u's', 'final': u'un'},
                                  u'xie': {'initial': u'x', 'final': u'ie'},
                                  u'qiang': {'initial': u'q', 'final': u'iang'},
                                  u'shang': {'initial': u'sh', 'final': u'ang'},
                                  u'zhang': {'initial': u'zh', 'final': u'ang'},
                                  u'sui': {'initial': u's', 'final': u'ui'},
                                  u'kuo': {'initial': u'k', 'final': u'uo'},
                                  u'kun': {'initial': u'k', 'final': u'un'},
                                  u'zai': {'initial': u'z', 'final': u'ai'},
                                  u'ren': {'initial': u'r', 'final': u'en'},
                                  u'ruan': {'initial': u'r', 'final': u'uan'},
                                  u'pei': {'initial': u'p', 'final': u'ei'},
                                  u'cheng': {'initial': u'ch', 'final': u'eng'},
                                  u'lei': {'initial': u'l', 'final': u'ei'},
                                  u'lui': {'initial': u'l', 'final': u'ui'},
                                  u'xu': {'initial': u'x', 'final': u'u'},
                                  u'ri': {'initial': u'r', 'final': u'i'},
                                  u'lun': {'initial': u'l', 'final': u'un'},
                                  u'luo': {'initial': u'l', 'final': u'uo'},
                                  u'be': {'initial': u'b', 'final': u'e'},
                                  u'men': {'initial': u'm', 'final': u'en'},
                                  u'ba': {'initial': u'b', 'final': u'a'},
                                  u'sheng': {'initial': u'sh', 'final': u'eng'},
                                  u'cha': {'initial': u'ch', 'final': u'a'},
                                  u'wo': {'initial': u'w', 'final': u'o'},
                                  u'ju': {'initial': u'j', 'final': u'u'},
                                  u'neng': {'initial': u'n', 'final': u'eng'},
                                  u'bo': {'initial': u'b', 'final': u'o'},
                                  u'mei': {'initial': u'm', 'final': u'ei'},
                                  u'bi': {'initial': u'b', 'final': u'i'},
                                  u'tian': {'initial': u't', 'final': u'ian'},
                                  u'bu': {'initial': u'b', 'final': u'u'},
                                  u'ruo': {'initial': u'r', 'final': u'uo'},
                                  u'le': {'initial': u'l', 'final': u'e'},
                                  u'que': {'initial': u'q', 'final': u'ue'},
                                  u'ji': {'initial': u'j', 'final': u'i'},
                                  u'huang': {'initial': u'h', 'final': u'uang'},
                                  u'shi': {'initial': u'sh', 'final': u'i'},
                                  u'shuo': {'initial': u'sh', 'final': u'uo'},
                                  u'yan': {'initial': u'y', 'final': u'an'},
                                  u'qun': {'initial': u'q', 'final': u'un'},
                                  u'qing': {'initial': u'q', 'final': u'ing'},
                                  u'ming': {'initial': u'm', 'final': u'ing'},
                                  u'yue': {'initial': u'y', 'final': u'ue'},
                                  u'zeng': {'initial': u'z', 'final': u'eng'},
                                  u'jia': {'initial': u'j', 'final': u'ia'},
                                  u'wu': {'initial': u'w', 'final': u'u'},
                                  u'pang': {'initial': u'p', 'final': u'ang'},
                                  u'xue': {'initial': u'x', 'final': u'ue'},
                                  u'zheng': {'initial': u'zh', 'final': u'eng'},
                                  u'chou': {'initial': u'ch', 'final': u'ou'},
                                  u'yun': {'initial': u'y', 'final': u'un'},
                                  u'chang': {'initial': u'ch', 'final': u'ang'},
                                  u'zui': {'initial': u'z', 'final': u'ui'},
                                  u'luan': {'initial': u'l', 'final': u'uan'},
                                  u'wei': {'initial': u'w', 'final': u'ei'},
                                  u'shui': {'initial': u'sh', 'final': u'ui'},
                                  u'ding': {'initial': u'd', 'final': u'ing'},
                                  u'zao': {'initial': u'z', 'final': u'ao'},
                                  u'ci': {'initial': u'c', 'final': u'i'},
                                  u'xi': {'initial': u'x', 'final': u'i'},
                                  u'jian': {'initial': u'j', 'final': u'ian'},
                                  u'jiao': {'initial': u'j', 'final': u'iao'},
                                  u'guan': {'initial': u'g', 'final': u'uan'},
                                  u'pa': {'initial': u'p', 'final': u'a'},
                                  u'guai': {'initial': u'g', 'final': u'uai'},
                                  u'wen': {'initial': u'w', 'final': u'en'},
                                  u'ran': {'initial': u'r', 'final': u'an'},
                                  u'qin': {'initial': u'q', 'final': u'in'},
                                  u'xin': {'initial': u'x', 'final': u'in'},
                                  u'dong': {'initial': u'd', 'final': u'ong'},
                                  u'yi': {'initial': u'y', 'final': u'i'},
                                  u'kuai': {'initial': u'k', 'final': u'uai'},
                                  u'tou': {'initial': u't', 'final': u'ou'},
                                  u'wan': {'initial': u'w', 'final': u'an'},
                                  u'cu': {'initial': u'c', 'final': u'u'},
                                  u'wang': {'initial': u'w', 'final': u'ang'},
                                  u'dui': {'initial': u'd', 'final': u'ui'},
                                  u'jing': {'initial': u'j', 'final': u'ing'},
                                  u'che': {'initial': u'ch', 'final': u'e'},
                                  u'wai': {'initial': u'w', 'final': u'ai'},
                                  u'duo': {'initial': u'd', 'final': u'uo'},
                                  u'fen': {'initial': u'f', 'final': u'en'},
                                  u'chi': {'initial': u'ch', 'final': u'i'},
                                  u'yin': {'initial': u'y', 'final': u'in'},
                                  u'rong': {'initial': u'r', 'final': u'ong'},
                                  u'yao': {'initial': u'y', 'final': u'ao'},
                                  u'long': {'initial': u'l', 'final': u'ong'},
                                  u'fei': {'initial': u'f', 'final': u'ei'},
                                  u'ting': {'initial': u't', 'final': u'ing'},
                                  u'qie': {'initial': u'q', 'final': u'ie'},
                                  u'nian': {'initial': u'n', 'final': u'ian'},
                                  u'lu': {'initial': u'l', 'final': u'u'},
                                  u'chu': {'initial': u'ch', 'final': u'u'},
                                  u'liang': {'initial': u'l', 'final': u'iang'},
                                  u'lou': {'initial': u'l', 'final': u'ou'},
                                  u'huan': {'initial': u'h', 'final': u'uan'},
                                  u'hen': {'initial': u'h', 'final': u'en'},
                                  u'man': {'initial': u'm', 'final': u'an'},
                                  u'kou': {'initial': u'k', 'final': u'ou'},
                                  u'ceng': {'initial': u'c', 'final': u'eng'},
                                  u'huai': {'initial': u'h', 'final': u'uai'},
                                  u'niang': {'initial': u'n', 'final': u'iang'},
                                  u'lang': {'initial': u'l', 'final': u'ang'},
                                  u'hua': {'initial': u'h', 'final': u'ua'},
                                  u'chun': {'initial': u'ch', 'final': u'un'},
                                  u'xian': {'initial': u'x', 'final': u'ian'},
                                  u'hun': {'initial': u'h', 'final': u'un'},
                                  u'huo': {'initial': u'h', 'final': u'uo'},
                                  u'jun': {'initial': u'j', 'final': u'un'},
                                  u'hui': {'initial': u'h', 'final': u'ui'},
                                  u'hu': {'initial': u'h', 'final': u'u'},
                                  u'gao': {'initial': u'g', 'final': u'ao'},
                                  u'gan': {'initial': u'g', 'final': u'an'},
                                  u'quan': {'initial': u'q', 'final': u'uan'},
                                  u'tan': {'initial': u't', 'final': u'an'},
                                  u'niu': {'initial': u'n', 'final': u'iu'},
                                  u'ling': {'initial': u'l', 'final': u'ing'},
                                  u'jiang': {'initial': u'j', 'final': u'iang'},
                                  u'biao': {'initial': u'b', 'final': u'iao'},
                                  u'chong': {'initial': u'ch', 'final': u'ong'},
                                  u'he': {'initial': u'h', 'final': u'e'},
                                  u'bei': {'initial': u'b', 'final': u'ei'},
                                  u'ben': {'initial': u'b', 'final': u'en'},
                                  u'tang': {'initial': u't', 'final': u'ang'},
                                  u'xun': {'initial': u'x', 'final': u'un'},
                                  u'mo': {'initial': u'm', 'final': u'o'},
                                  u'mi': {'initial': u'm', 'final': u'i'},
                                  u'po': {'initial': u'p', 'final': u'o'},
                                  u'lve': {'initial': u'l', 'final': u've'},
                                  u'cai': {'initial': u'c', 'final': u'ai'},
                                  u'pian': {'initial': u'p', 'final': u'ian'},
                                  u'can': {'initial': u'c', 'final': u'an'},
                                  u'ning': {'initial': u'n', 'final': u'ing'},
                                  u'dang': {'initial': u'd', 'final': u'ang'},
                                  u'cang': {'initial': u'c', 'final': u'ang'},
                                  u'sai': {'initial': u's', 'final': u'ai'},
                                  u'gong': {'initial': u'g', 'final': u'ong'},
                                  u'sao': {'initial': u's', 'final': u'ao'},
                                  u'san': {'initial': u's', 'final': u'an'},
                                  u'pin': {'initial': u'p', 'final': u'in'},
                                  u'ai': {'initial': u'', 'final': u'ai'},
                                  u'chai': {'initial': u'ch', 'final': u'ai'},
                                  u'chan': {'initial': u'ch', 'final': u'an'},
                                  u'xiao': {'initial': u'x', 'final': u'iao'},
                                  u'bian': {'initial': u'b', 'final': u'ian'},
                                  u'an': {'initial': u'', 'final': u'an'},
                                  u'duan': {'initial': u'd', 'final': u'uan'},
                                  u'zhi': {'initial': u'zh', 'final': u'i'},
                                  u'cong': {'initial': u'c', 'final': u'ong'},
                                  u'qiu': {'initial': u'q', 'final': u'iu'},
                                  u'zou': {'initial': u'z', 'final': u'ou'},
                                  u'pong': {'initial': u'p', 'final': u'ong'},
                                  u'tai': {'initial': u't', 'final': u'ai'},
                                  u'tuo': {'initial': u't', 'final': u'uo'},
                                  u'zhun': {'initial': u'zh', 'final': u'un'},
                                  u'ni': {'initial': u'n', 'final': u'i'},
                                  u'hong': {'initial': u'h', 'final': u'ong'},
                                  u'shuang': {'initial': u'sh', 'final': u'uang'},
                                  u'min': {'initial': u'm', 'final': u'in'},
                                  u'na': {'initial': u'n', 'final': u'a'},
                                  u'juan': {'initial': u'j', 'final': u'uan'},
                                  u'mian': {'initial': u'm', 'final': u'ian'},
                                  u'nan': {'initial': u'n', 'final': u'an'},
                                  u'si': {'initial': u's', 'final': u'i'},
                                  u'liu': {'initial': u'l', 'final': u'iu'},
                                  u'hou': {'initial': u'h', 'final': u'ou'},
                                  u'zhuo': {'initial': u'zh', 'final': u'uo'},
                                  u'you': {'initial': u'y', 'final': u'ou'},
                                  u'shan': {'initial': u'sh', 'final': u'an'},
                                  u'nu': {'initial': u'n', 'final': u'u'},
                                  u'nv': {'initial': u'n', 'final': u'v'},
                                  u'pao': {'initial': u'p', 'final': u'ao'},
                                  u'lan': {'initial': u'l', 'final': u'an'},
                                  u'bie': {'initial': u'b', 'final': u'ie'},
                                  u'fu': {'initial': u'f', 'final': u'u'},
                                  u'zhou': {'initial': u'zh', 'final': u'ou'},
                                  u'ru': {'initial': u'r', 'final': u'u'},
                                  u'kai': {'initial': u'k', 'final': u'ai'},
                                  u'hang': {'initial': u'h', 'final': u'ang'},
                                  u'kan': {'initial': u'k', 'final': u'an'},
                                  u'fa': {'initial': u'f', 'final': u'a'},
                                  u'zhe': {'initial': u'zh', 'final': u'e'},
                                  u'zhuan': {'initial': u'zh', 'final': u'uan'},
                                  u'mai': {'initial': u'm', 'final': u'ai'},
                                  u'song': {'initial': u's', 'final': u'ong'},
                                  u'yuan': {'initial': u'y', 'final': u'uan'},
                                  u'xiang': {'initial': u'x', 'final': u'iang'},
                                  u'mao': {'initial': u'm', 'final': u'ao'},
                                  u'fo': {'initial': u'f', 'final': u'o'},
                                  u'a': {'initial': u'', 'final': u'a'},
                                  u'e': {'initial': u'', 'final': u'e'},
                                  u'ke': {'initial': u'k', 'final': u'e'},
                                  u'kuang': {'initial': u'k', 'final': u'uang'},
                                  u'bing': {'initial': u'b', 'final': u'ing'},
                                  u'xia': {'initial': u'x', 'final': u'ia'},
                                  u'zuo': {'initial': u'z', 'final': u'uo'},
                                  u'de': {'initial': u'd', 'final': u'e'},
                                  u'yang': {'initial': u'y', 'final': u'ang'},
                                  u'ku': {'initial': u'k', 'final': u'u'},
                                  u'zhong': {'initial': u'zh', 'final': u'ong'},
                                  u'qi': {'initial': u'q', 'final': u'i'},
                                  u'deng': {'initial': u'd', 'final': u'eng'},
                                  u'lin': {'initial': u'l', 'final': u'in'},
                                  u'kao': {'initial': u'k', 'final': u'ao'},
                                  u'ao': {'initial': u'', 'final': u'ao'},
                                  u'qiong': {'initial': u'q', 'final': u'iong'},
                                  u'wa': {'initial': u'w', 'final': u'a'},
                                  u'lie': {'initial': u'l', 'final': u'ie'},
                                  u'ne': {'initial': u'n', 'final': u'e'},
                                  u'kang': {'initial': u'k', 'final': u'ang'},
                                  u'chuan': {'initial': u'ch', 'final': u'uan'},
                                  u'zu': {'initial': u'z', 'final': u'u'},
                                  u'zhan': {'initial': u'zh', 'final': u'an'},
                                  u'sha': {'initial': u'sh', 'final': u'a'},
                                  u'tun': {'initial': u't', 'final': u'un'},
                                  u'gou': {'initial': u'g', 'final': u'ou'},
                                  u'pan': {'initial': u'p', 'final': u'an'},
                                  u'chao': {'initial': u'ch', 'final': u'ao'},
                                  u'zun': {'initial': u'z', 'final': u'un'},
                                  u'nai': {'initial': u'n', 'final': u'ai'},
                                  u'ou': {'initial': u'', 'final': u'ou'},
                                  u'gun': {'initial': u'g', 'final': u'un'},
                                  u'zha': {'initial': u'zh', 'final': u'a'},
                                  u'ha': {'initial': u'h', 'final': u'a'},
                                  u'nao': {'initial': u'n', 'final': u'ao'},
                                  u'zan': {'initial': u'z', 'final': u'an'},
                                  u'kua': {'initial': u'k', 'final': u'ua'},
                                  u'me': {'initial': u'm', 'final': u'e'},
                                  u'shua': {'initial': u'sh', 'final': u'ua'},
                                  u'la': {'initial': u'l', 'final': u'a'},
                                  u'peng': {'initial': u'p', 'final': u'eng'},
                                  u'zhui': {'initial': u'zh', 'final': u'ui'},
                                  u'tie': {'initial': u't', 'final': u'ie'},
                                  u'qiao': {'initial': u'q', 'final': u'iao'},
                                  u'gai': {'initial': u'g', 'final': u'ai'},
                                  u'cao': {'initial': u'c', 'final': u'ao'},
                                  u'mou': {'initial': u'm', 'final': u'ou'},
                                  u'shun': {'initial': u'sh', 'final': u'un'},
                                  u'pi': {'initial': u'p', 'final': u'i'},
                                  u'zhai': {'initial': u'zh', 'final': u'ai'},
                                  u'gang': {'initial': u'g', 'final': u'ang'},
                                  u'suan': {'initial': u's', 'final': u'uan'},
                                  u'shuai': {'initial': u'sh', 'final': u'uai'},
                                  u'su': {'initial': u's', 'final': u'u'},
                                  u're': {'initial': u'r', 'final': u'e'},
                                  u'tui': {'initial': u't', 'final': u'ui'},
                                  u'we': {'initial': u'w', 'final': u'e'},
                                  u'tao': {'initial': u't', 'final': u'ao'},
                                  u'tuan': {'initial': u't', 'final': u'uan'},
                                  u'diao': {'initial': u'd', 'final': u'iao'},
                                  u'dou': {'initial': u'd', 'final': u'ou'},
                                  u'se': {'initial': u's', 'final': u'e'},
                                  u'jue': {'initial': u'j', 'final': u'ue'},
                                  u'chui': {'initial': u'ch', 'final': u'ui'},
                                  u'pai': {'initial': u'p', 'final': u'ai'},
                                  u'mang': {'initial': u'm', 'final': u'ang'},
                                  u'teng': {'initial': u't', 'final': u'eng'},
                                  u'qia': {'initial': u'q', 'final': u'ia'},
                                  u'ong': {'initial': u'', 'final': u'ong'},
                                  u'o': {'initial': u'', 'final': u'o'},
                                  u'nei': {'initial': u'n', 'final': u'ei'},
                                  u'miao': {'initial': u'm', 'final': u'iao'},
                                  u'niao': {'initial': u'n', 'final': u'iao'},
                                  u'sa': {'initial': u's', 'final': u'a'},
                                  u'rui': {'initial': u'r', 'final': u'ui'},
                                  u'cuo': {'initial': u'c', 'final': u'uo'},
                                  u'ce': {'initial': u'c', 'final': u'e'},
                                  u'leng': {'initial': u'l', 'final': u'eng'},
                                  u'geng': {'initial': u'g', 'final': u'eng'},
                                  u'nang': {'initial': u'n', 'final': u'ang'},
                                  u'zhua': {'initial': u'zh', 'final': u'ua'},
                                  u'rang': {'initial': u'r', 'final': u'ang'},
                                  u'kuan': {'initial': u'k', 'final': u'uan'},
                                  u'cuan': {'initial': u'c', 'final': u'uan'},
                                  u'piao': {'initial': u'p', 'final': u'iao'},
                                  u'pie': {'initial': u'p', 'final': u'ie'},
                                  u'nuan': {'initial': u'n', 'final': u'uan'},
                                  u'nong': {'initial': u'n', 'final': u'ong'},
                                  u'shai': {'initial': u'sh', 'final': u'ai'},
                                  u'dun': {'initial': u'd', 'final': u'un'},
                                  u'diu': {'initial': u'd', 'final': u'iu'},
                                  u'cou': {'initial': u'c', 'final': u'ou'}}

dic_initial_2_sampa = {u'':u'',
                       u'r':u"r\\'",
                       u'g':u'k',
                       u'f':u'f',
                       u'h':u'x',
                       u'm':u'm',
                       u'l':u'l',
                       u'n':u'n',
                       u'w':u'w',
                       u'y':u'j',

                       u's':u'c',
                       u'sh':u'c',
                       u't':u'c',
                        u'q':u'c',
                        u'k':u'c',
                       u'j':u'c',
                       u'c':u'c',
                       u'b':u'c',
                       u'p':u'c',
                       u'd':u'c',
                       u'ch':u'c',
                       u'zh':u'c',
                       u'x':u'c',
                       u'z':u'c'}

dic_final_2_sampa = {u'en':[u'@n',u'n'],
                     u'ei':[u'eI^',u'i'],
                     u've':[u'H',u'9'],
                     u'ai':[u'aI^',u'i'],
                     u'uan':[u'w',u'an',u'n'],
                     u'iu':[u'j',u'oU^',u'u'],
                     u'ong':[u'UN',u'N'],
                     u'ao':[u'AU^',u'u'],
                     u'an':[u'an',u'n'],
                     u'uai':[u'w',u'aI^',u'i'],
                     u'ang':[u'AN',u'N'],
                     u'iong':[u'j',u'UN',u'N'],
                     u'in':[u'in',u'n'],
                     u'ia':[u'j',u'a'],
                     u'ing':[u'iN',u'N'],
                     u'ie':[u'j',u'E'],
                     u'er':[u'@',u"r\\'"],
                     u'iao':[u'j',u'AU^',u'u'],
                     u'ian':[u'j',u'En',u'n'],
                     u'eng':[u'7N',u'N'],
                     u'iang':[u'j',u'AN',u'N'],
                     u'ui':[u'w',u'eI^',u'i'],
                     u'uang':[u'w',u'AN',u'N'],
                     u'a':[u'a'],
                     u'e':[u'7'],
                     u'i':[u'i'],
                     u'o':[u'O'],
                     u'uo':[u'w',u'O'],
                     u'un':[u'w',u'@n',u'n'],
                     u'u':[u'u'],
                     u'v':[u'y'],
                     u'ue':[u'H',u'9'],
                     u'ou':[u'oU^',u'u'],
                     u'ua':[u'w',u'a']}

non_pinyin = [u'\uff08']

if __name__ == '__main__':
    initials = []
    finals   = []
    for v in dic_pinyin_2_initial_final_map.values():
        initials.append(v['initial'])
        finals.append(v['final'])

    print len(set(initials))
    print len(set(finals))

