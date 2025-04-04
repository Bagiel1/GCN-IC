x= [[1,4,5,6,3,2], [1,4,6,10,3,5], [1,4,2,3,9,8], [1,2,3,6,8,10]]
z= [1,4,5,6,3,10]
p= [1,4,5,6,3,10]

edge= []

def jaccard(x, top_k):
    for i in range(len(x)):
        for j in range(len(x)):
            b= len(set(x[i][:top_k]) & set(x[j][:top_k])) / len(set(x[i][:top_k]) | set(x[j][:top_k]))
            if b >= 0.3:
                edge.append([i,j])
    return edge

print(jaccard(x, 6))

def compute_rbo(X, top_k):
    
    edges= []
    r = 0.9

    for z in range(len(X)):
        for p in range(len(X)):

            stored = set()  
            acum_inter = 0
            score = 0
            img1_leftover = set()
            img2_leftover = set()

            for i in range(top_k):
                img1_elm = x[z][i]
                img2_elm = x[p][i]
                if img1_elm not in stored and img1_elm == img2_elm:
                    acum_inter += 1
                    stored.add(img1_elm)
                else:
                    if img1_elm not in stored:
                        if img1_elm in img2_leftover:
                            
                            acum_inter += 1
                            stored.add(img1_elm)
                            img2_leftover.remove(img1_elm)
                            
                        else:
                            img1_leftover.add(img1_elm)

                    if img2_elm not in stored:
                        if img2_elm in img1_leftover:
                           
                            acum_inter += 1
                            stored.add(img2_elm)
                            img1_leftover.remove(img2_elm)
                        else:
                            img2_leftover.add(img2_elm)

                score += (r**((i+1) - 1)) * (acum_inter / (i+1))
            scrN= (1-r) * score
            if scrN >= 0.3:
                edges.append([z,p])

    return edges


print(compute_rbo(x,6))