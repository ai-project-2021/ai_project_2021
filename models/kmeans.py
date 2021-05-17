import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from utils import get_product_recommend, get_rfm_data
from tqdm import tqdm, trange


def data_max(s):
    return np.max(s)

def d_tool(a, b):
    return np.sum((a - b) ** 2, axis=1) ** 0.5
    # return np.sum(np.power((a - b), 2)) ** 0.5

class Recommedation:
    def __init__(self):
        self.data = get_rfm_data(get_product_recommend())
        # self.data["R_Value"] = ( self.data["R_Value"] - self.data["R_Value"].mean() ) / self.data["R_Value"].std()
        # self.data["F_Value"] = ( self.data["F_Value"] - self.data["F_Value"].mean() ) / self.data["F_Value"].std()
        # self.data["M_Value"] = ( self.data["M_Value"] - self.data["M_Value"].mean() ) / self.data["M_Value"].std()
        self.rfm = self.data[["R_Value", "F_Value", "M_Value"]]

    def package_kmeans(self):
        rfm = self.data
        
        # package를 이용한 kmeans
        # print(rfm.columns)
        X = rfm[["R_Value", "F_Value", "M_Value"]] # rfm.drop(['Customer Id'])
        k = KMeans(n_clusters=3, random_state=0).fit(X)
        fig = px.scatter_3d(
            rfm, x="R_Value", y="F_Value", z="M_Value", color = k.labels_
        )
        
        # -----------
        print("Label value",k.labels_)
        print("\n")
        print("cluster center", k.cluster_centers_)
        fig.show()
        
        # return fig

    def predict(self, k): # K = 3
        rfm = self.data 
        
        c_num = rfm["Customer Id"].nunique() # number of customers
        
        # centroid 생성
        centroid = random.sample(range(rfm["Customer Id"].min(), rfm["Customer Id"].max(), k)) # k개의 customer id
        
        # distance 
        distance = np.zeros(c_num, c_num) # np.zeros(c_num, k)
        label = np.zeros(c_num)

        # c_num != k
        # c_num : All
        
        # i : Train Case    rfm["Customer Id"] 80%
        # j : Test Case     rfm["Customer Id"] 20%
        for i in range(c_num):
            distance[i] = d_tool(centroid, rfm[i])
            #rfm["Customer Id"][centroid], rfm[i]
            
            # for j in range(k):
            #     distance[j][i] = d_tool(rfm[j][:], rfm[i][:])
            label[i] = distance[i].argsort()[0]
            # k_means_idx = distance[i].argsort()[0]
            # cluster = np.mean(distance[i][k_means])    
            # cluster = np.argmin(distance[i]) # 제일 가까운 애들의 평균?
            
        # Cluster 갱신
        # Centroid 재계산.
    
    # https://owenit.tistory.com/1
    def sample(self, n, k) : 
        # n = rfm[["R_Value", "F_Value", "M_Value"]].shape (3, records)
        print(n.shape)
        d = n.shape[1]
        c = np.random.choice(range(len(n)), k)
        c = n[c].astype('float32')
        
        init = True
        while init : 
            print("RESTART")
            cluster = np.zeros((len(n), k))
            labels = np.zeros((len(n)))
            
            for i in range(len(n)) : # Customer ID
                temp = np.zeros((k))
                for j in range(k) : 
                    temp[j] = np.sum((c[j] - n[i]) ** 2) ** 0.5
                cluster[i][temp.argmin()] = 1 
                labels[i] = temp.argmin()
            
            cnt = 0
            for p in range(k) : 
                clu_vec = n[np.where(cluster[:, p] == 1)]
                k_to_c = np.sum(clu_vec, axis=0) / len(clu_vec) 
                
                point_c = np.sum((c[p].reshape(-1) - k_to_c.reshape(-1)) ** 2) ** 0.5
                    
                if point_c < 3 : # IF 문의 Condition이 왜 이렇게 되는지?
                    cnt+=1 
                    
                    if cnt == k:
                        init = False
                        break
                    
                c[p] = k_to_c
            print(c)
                
        return cluster, c, labels


    def hdbscan(self, mn): #mn : mininum number of neighbors
        

                
if __name__ == "__main__" : 
    model = Recommedation()
    model.package_kmeans()
    clu, c, labels = model.sample(model.rfm.values, 3)
    print("CLU")
    print(clu)
    print("C")
    print(c)
    print("labels")
    print(labels)
    
    fig = px.scatter_3d(
        model.rfm, x="R_Value", y="F_Value", z="M_Value", color = labels
    )
    
    fig.show()