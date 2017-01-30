# -*- coding: utf-8 -*-

import Levenshtein

string1 = "RELLY 11OREILD ORE\nRESTful Webサービス R\nービス. daw-,一 ELL\n初めてのPerl\n版\nTim Pturnix\n19\nPUnP\n演習ス\n?習サーバサイド用\nmスクリプト言語\n片山幸雄\n30分でつくれるAndroidアプリ-?鞆200m-,, 7\nOldアプリApp Inventorで\nはじめよう!\nクラウド活用のためのAndroid業務アプリ開発入門出村成和?-\n学生のためのPIP Et=蕭a0空ブシKh4N\nたにぐちまaJA is!\nよくわかるPHPの教科書\nAndroidアプリがWebブラウザ上で部品を並べるだけでできあがる\njQuery Mobileはスマホ開発を\n制作入\n簡単に美しく!!\n| Android Cookbook-巽の謂)T:-|\nアプリの\n価値を高める\nCookbook/黠テクニック\n柴田文ethasai tu.. ,藤枝崇史,\n初歩からわかる\n●安生真\nAndroid最新プログラ,ミング(IF\nアプリケーション 鈴木哲哉+8-LLi\nndrol\nー\n逆引き\nSDK\nDR ハンドフック\nAndroid 2.3/2.2/21/20/\n1.6/1.5凭ージョンに対E-\nテクニック\niPhone\nショッ開発ガイドHTNLrassgJeudept\n幸軽マィコン率ドArduino圭サで計測&解析@ileae \" coaai\n大川善邦著 CQ\nEUti\n"

string2 = "RELLY OREILD ORE\nRESTful Webサービス R\nービス. daw-,一 ELL\n初めてのPerl\n版\nTim Pturnix\n19\nPUnP\n演習ス\n?習サーバサイド用\nmスクリプト言語\n片山幸雄\n30分でつくれるAndroidアプリ-?鞆200m-,, 7\nOldアプリApp Inventorで\nはじめよう!\nクラウド活用のためのAndroid業務アプリ開発入門出村成和?-\n学生のためのPIP Et=蕭a0空ブシKh4N\nたにぐちまaJA is!\nよくわかるPHPの教科書\nAndroidアプリがWebブラウザ上で部品を並べるだけでできあがる\njQuery Mobileはスマホ開発を\n制作入\n簡単に美しく!!\n| Android Cookbook-巽の謂)T:-|\nアプリの\n価値を高める\nCookbook/黠テクニック\n柴田文ethasai tu.. ,藤枝崇史,\n初歩からわかる\n●安生真\nAndroid最新プログラ,ミング(IF\nアプリケーション 鈴木哲哉+8-LLi\nndrol\nー\n逆引き\nSDK\nDR ハンドフック\nAndroid 2.3/2.2/21/20/\n1.6/1.5凭ージョンに対E-\nテクニック\niPhone\nショッ開発ガイドHTNLrassgJeudept\n幸軽マィコン率ドArduino圭サで計測&解析@ileae \" coaai\n大川善邦著 CQ\nEUti\n"

# string1 = string1.decode('utf-8')
# string2 = string2.decode('utf-8')

print(Levenshtein.distance(string1, string2))