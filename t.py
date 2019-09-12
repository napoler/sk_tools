from sk_tools import TMNB
t= TMNB()
data_array=[{'data':"柯基犬是个狗子",'class':1},{'data':"哈士奇是个狗子",'class':10},{'data':"奇是个狗子",'class':120}]
# 训练
t.fit_class_list(data_array)


data=[ "哈士奇是个狗","奇是个狗子"]


# tfidfVectorizer = TfidfVectorizer()
# train_term_matrix = tfidfVectorizer.transform(data)
# print(train_term_matrix)
print("预测结果:",t.predict(data))

# !ls