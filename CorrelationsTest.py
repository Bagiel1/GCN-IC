x= [[1,40,50,60,30,10], [0,2,14,5,3,10], [2,4,6,1,3,5], [0,1,20,37,99,8], [4,65,1,6,8,10]]

edge= []

def jaccard(x, top_k):
    for i in range(len(x)):
        for j in range(len(x)):
            b= len(set(x[i][:top_k]) & set(x[j][:top_k])) / len(set(x[i][:top_k]) | set(x[j][:top_k]))
            if b >= 0.3:
                edge.append([i,j])
    return edge

print(jaccard(x, 6))