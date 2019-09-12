from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle


class TerryMultinomialNB:
  """
  分类训练预测类
  """
  def __init__(self):
    self.data = []
    self.vocabulary_path="model/vocabulary"
    self.model_path= 'model/train_model.model'
  def fit_class_list(self,data_array):
    '''
    输入格式
    class_list=[{'data':"柯基犬是个狗子",'class':"柯基犬"},
    {'data':"哈士奇是个狗子",'class':"哈士奇"}
    ]

    '''
    x_list,target_list=self.bulid_data(data_array)
    # print(x_list)
    # print(target_list)
    # 实例化CountVectorizer
    train_tfidfVectorizer = TfidfVectorizer()

    train_term_matrix = train_tfidfVectorizer.fit_transform(x_list)
    vocabulary = train_tfidfVectorizer.vocabulary_
    
    clf = MultinomialNB().fit(train_term_matrix, target_list)
# #     这里保存模型
    joblib.dump(clf, self.model_path)
    pickle.dump(vocabulary, open(self.vocabulary_path, "wb"))
    print('已经保存')
  
  def bulid_data(self,data_array):
    """
        输入格式
    data_array=[{'data':"柯基犬是个狗子",'class':"柯基犬"},
    {'data':"哈士奇是个狗子",'class':"哈士奇"}
    ]
    """
    x_list=[]
    target_list=[]
    for item in data_array:
      x_list.append(item['data'])
      target_list.append(item['class'])
    
      
    return  x_list,target_list
  def load(self):
    return joblib.load(self.model_path)
    
    return
  def predict(self,x_list):
    model=self.load()
    print(x_list)
    # 实例化CountVectorizer
    vocabulary = pickle.load(open(self.vocabulary_path, "rb"))
    tfidfVectorizer = TfidfVectorizer(vocabulary=vocabulary)
    p_term_matrix = tfidfVectorizer.fit_transform(x_list)
    doc_class_predicted = model.predict(p_term_matrix)
#     print('yc',doc_class_predicted)
#     返回预测结果
    return doc_class_predicted

  

"""
#使用示例

from sk_tools import TMNB
t= TMNB()


# 训练示例


data_array=[{'data':"柯基犬是个狗子",'class':1},{'data':"哈士奇是个狗子",'class':10},{'data':"奇是个狗子",'class':120}]
# 训练
t.fit_class_list(data_array)

# 预测示例

data=[ "哈士奇是个狗","奇是个狗子"]
print("预测结果:",t.predict(data))

"""