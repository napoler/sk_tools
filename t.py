from sk_tools import TMNB
import sqlite3

DB='data/xigua.db'
def get_list():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # sql="SELECT * FROM xg_links ORDER BY RANDOM() limit 1000"
    sql="SELECT * FROM xg_links"
    c.execute(sql)
    all = c.fetchall()
    # print(one)

    conn.close()
    return all
# print("all",get_list())
def build_classification(num):
    if(num<=100):
        return '0'
    elif(num<1000):
        return '1000'
    elif(num<10000):
        return '1w'
    elif(num<100000):
        return '10w'
    else:
        return 'max'
data_array=[]

for item in get_list():
    tclass = build_classification(item[3]) #基于播放量
    data_array.append({'data':item[2],'class':tclass})







print(data_array)







t= TMNB()
# data_array=[{'data':"柯基犬是个狗子",'class':1},{'data':"哈士奇是个狗子",'class':10},{'data':"奇是个狗子",'class':120}]
# 训练
# t.fit_class_list(data_array)


data=[ "哈士奇是个狗","牛逼","元组可以使用下标索引来访问元组中的值，如下实例:","VIP小岛戏水，老王的花式狗刨让老婆嘲笑，你们的泳姿优美吗","张无忌与武当派相认，外公白眉鹰王得知也是开心自豪无比，太开心"]


# tfidfVectorizer = TfidfVectorizer()
# train_term_matrix = tfidfVectorizer.transform(data)
# print(train_term_matrix)
print("预测结果:",t.predict(data))

# !ls