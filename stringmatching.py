'''
    File name: semanticmatching.py
    Author: Hugo Haggren, Leo Hatvani
'''

import os
import numpy as np
import math
from fuzzywuzzy import fuzz
from nltk import RegexpTokenizer
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Levenshtein.StringMatcher import StringMatcher
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

data_path = 'data'


def load_data():
    # Loads test cases and test case names
    # 
    # Out: 
    # doc_labels - Names of all test case documents
    # data - The raw strings of all test case documents
    doc_labels = []
    data = []
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if (file[0] == "."):
                continue
            doc_labels.append(subdir.replace(data_path, '').replace('/','\\') + '\\' + file)
            f = open(os.path.join(subdir, file),'r',encoding='utf-8')
            data.append(f.read()) 
    return doc_labels, data

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def levenshtein_ratio(s1, s2):
    m = StringMatcher(None, s1, s2)
    return truncate(m.ratio(), 2), m.distance()



def fuzzy_match(print_mode=False):
    doc_labels, data = load_data()
    link_list = check_for_links(doc_labels, data)
    N = len(data)
    fuzzy_sim = np.zeros((N,N))
    lev_dist = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            score, dist = levenshtein_ratio(data[i],data[j])
            fuzzy_sim[i,j] = score
            fuzzy_sim[j,i] = score
            lev_dist[i,j] = dist
            lev_dist[j,i] = dist
            if score >= 0.8 and i != j and print_mode:
                print(int(link_list[i]),';', score,';', doc_labels[i][:33],';', doc_labels[j][:33])
    result = pd.DataFrame(fuzzy_sim, columns=doc_labels)
    distance = pd.DataFrame(lev_dist, columns=doc_labels)
    return result, distance

def wordwise_match():
    doc_labels, data = load_data()
    N = len(data)
    word_sim = np.zeros((N,N))

    tokenized_data = []
    tokenizer = RegexpTokenizer('\w+')

    for i,d in enumerate(data):
        tokenized_data.append(tokenizer.tokenize(d))    
 
    for i in range(N):
        for j in range(N):
            #This version uses simple list operations, fuzzwuzzy might be better but I suspect that it will be slower
            intersection = len(set(tokenized_data[i])&set(tokenized_data[j]))
            union = len(set(tokenized_data[i])) + len(set(tokenized_data[j])) - intersection
            word_sim[i,j] = intersection / union
    
    result = pd.DataFrame(word_sim, columns=doc_labels)
    print(result)
    return result

def visualize(param):
    infile = open('newfuzzysim.pickle','rb')
    distfile = open('distances.pickle','rb')
    df = pickle.load(infile)
    df_dist = pickle.load(distfile)
    values = df.values
    dist_values = df_dist.values

    upper_half = values[np.triu_indices(len(values), k=1)]
    dist_uh = dist_values[np.triu_indices(len(dist_values), k=1)]
    #plt.style.use('ggplot')

    if param == 'histo':
        top_cases = []
        distances = np.zeros(11)
        for i,val in enumerate(upper_half):
            if val//10 >= 9:
                top_cases.append(val)
                
                if val == 100:
                    distances[10] = distances[10] + dist_uh[i]
                else:
                    idx = int(val % 10)
                    distances[idx] = distances[idx] + dist_uh[i]
        
        print(top_cases)
        #print(distances)
        bins = np.histogram(top_cases, bins=11)
        print(distances / bins[0])
        plt.hist(top_cases, bins=11,edgecolor='black',linewidth=1)

    if param == 'histo20':
        top_cases = []
        
        for val in upper_half:
            if val//10 >= 8:
                top_cases.append(val)
                
        plt.hist(upper_half, bins=50,edgecolor='black',linewidth=1)
        #plt.xticks([80,85,90,95,100])

    if param == 'bar':
        bags = [0, 0, 0, 0]
        for val in upper_half:
            if val//10 == 8:
                bags[0] = bags[0] + 1 
            if val//10 >= 9:
                if val == 100:
                    bags[3] = bags[3] + 1
                #elif val == 99:
                    #bags[3] = bags[3] + 1    
                elif val % 10 > 4:
                    bags[2] = bags[2] + 1
                elif val == 90:
                    bags[0] = bags[0] + 1 
                else:
                    bags[1] = bags[1] + 1 
        #plt.bar(x=[1, 2, 3, 4], height = bags, width = 0.5, tick_label=['80-89','90-95','96-99','100'], color=['r','#ffa500','y','g'], edgecolor='black',linewidth=1.2)
        plt.bar(x=1, height = bags[0], width = 0.5, color='r', edgecolor='black',linewidth=1)
        plt.bar(x=2, height = bags[1], width = 0.5, color='#ffa500', edgecolor='black',linewidth=1)
        #plt.bar(x=3, height = bags[2], width = 0.5, color='#fce803', edgecolor='black',linewidth=1)
        plt.bar(x=3, height = bags[2], width = 0.5, color='y', edgecolor='black',linewidth=1)
        plt.bar(x=4, height = bags[3], width = 0.5, color='g', edgecolor='black',linewidth=1)
        for i, v in enumerate(bags):
            plt.text( i + .95 , v + 1 , str(v), color='black')
        plt.xticks([1, 2, 3, 4],['[0.80,0.90]','[0.91,0.94]','[0.95,0.99]','1.00'])
        plt.legend(labels=['Partially Similar','Similar','Very Similar','Identical'],ncol= 4,bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
        
    
    plt.xlabel('Levenshtein Ratio')
    plt.ylabel('# of Test Case Pairs')
    plt.show()

def check_for_links(doc_labels, data):
    key_words = ['1/1524', 'ref', '131 32-']
    output = np.zeros(len(data))
    for i,doc in enumerate(data):
        for k in key_words:
            if k in doc:
                #print(k, doc_labels[i])    
                output[i] = 1
                

    return output

def classify():
    # Divides the test-case pairs into four classes based on their levenshtein ratio

    doc_labels, data = load_data()
    infile = open('newfuzzysim.pickle','rb')
    df = pickle.load(infile)
    values = df.values
    classes = []
    counted = []
    prod_dict = {
                '\\1524-KRC 161 410': [],
                '\\1524-KRC 161 490': [],
                '\\1524-KRC 161 496': [],
                '\\1524-KRC 161 469': [],
                '\\1524-KRC_118_40_': []
                }
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            doc_i = doc_labels[i][:33]
            doc_j = doc_labels[j][:33]
            val = values[i,j]
            #print(val)
            c = ''
            if val//10 == 8:
                c = 'PS'
                classes.append([doc_i, doc_j, c])
                if doc_i not in counted:
                    prod_dict[doc_i[:17]].append(c)
                    counted.append(doc_i)
                if doc_j not in counted:
                    prod_dict[doc_j[:17]].append(c)
                    counted.append(doc_j)
            if val//10 >= 9:
                if val == 100:
                    c = 'I'
                #elif val == 99:
                #    c = 'I'
                elif val % 10 > 1:
                    c = 'VS'
                elif val == 90:
                    c = 'VS'
                else:
                    c = 'S'
                classes.append([doc_i, doc_j, c])
                if doc_i not in counted:
                    prod_dict[doc_i[:17]].append(c)
                    counted.append(doc_i)
                if doc_j not in counted:
                    prod_dict[doc_j[:17]].append(c)
                    counted.append(doc_j)
    result = pd.DataFrame(np.array(classes))
    #result.to_excel("classified_TCpairs.xlsx")
    #visualize_dict(prod_dict)
    return result
            
def visualize_dict(dict):
    labels = ['RBS1','RBS2','RBS3','RBS4','RBS5']
    category_names = ['Identical','Very Similar','Similar','Partially Similar']
    short_names = ['I','VS','S','PS']
    category_colors = ['g','y','#ffa500','r']
    idx = 0
    vals = []
    for key in dict:
       freq = CountFrequency(dict[key])
       tmp = []
       for n in short_names:
           if n in freq:
               tmp.append(freq[n])
           else:
               tmp.append(0)
       vals.append(tmp)

    data = np.array(vals)
    data_cum = data.cumsum(axis=1)
    #category_colors = plt.get_cmap('RdBu')(
    #    np.linspace(0.15, 0.85, data.shape[1]))
    #plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(9.2, 5))
    #ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        if i == 3:
            starts[4] = starts[4] + 21
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color, edgecolor='black')
        xcenters = starts + widths / 2

        #r, g, b, _ = color
        text_color = 'white' #if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if not c == 0:
                ax.text(x, y, str(int(c)), ha='center', va='center',
                        color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    #plt.hist(vals, label=labels, orientation='horizontal', stacked=True)
    #plt.legend(loc='upper right')
    plt.show()

def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    
    for key, value in freq.items(): 
        print (f"{key} : {value}")
    return freq

def evaluate():
    classes = classify()
    cval = classes.values
    df = pd.read_excel(r'excelfiles/evaluations.xlsx')
    dval = df.values
    y_true = dval[:,2]
    y_pred = cval[:100,2]
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    N = 155
    p_avg = 0
    r_avg = 0
    f_avg = 0
    for i in range(N):

        inds = random_undersample(df)

        y_t = y_true[inds]
        y_p = y_pred[inds]
        p, r, f1, supp = precision_recall_fscore_support(y_t, y_p, labels=['PS','S','VS','I'])
        p_avg = p_avg + p/N
        r_avg = r_avg + r/N
        f_avg = f_avg + f1/N

    plt.hist(y_t, bins=4,edgecolor='black',linewidth=1)
    plt.show()
    #print(y_t.shape, y_p.shape)
    #p, r, f1, supp = precision_recall_fscore_support(y_t, y_p, labels=['PS','S','VS','I'])
    return p_avg, r_avg, f_avg, acc, bacc

def random_undersample(c):
    vals = c.values
    X = vals[:98,:2]
    y = vals[:100,2]
    #np.delete(X, [75, 92], axis=0)
    #print(y[75], y[92])
    y = np.delete(y, [75, 92])
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X,y)
    inds = rus.sample_indices_
    return inds

def cheeky_test():
    classes = classify()
    df = pd.read_excel(r'excelfiles/evaluations.xlsx')
    cval = classes.values
    dval = df.values
    for i in range(len(dval)):
        print(cval[i,:])
        print(dval[i,:])
        print(i)

    y_true = dval[:,2]
    y_pred = cval[:100,2]
    y_t = np.delete(y_true, [75, 92])
    y_p = np.delete(y_pred, [75, 92])
    p, r, f1, supp = precision_recall_fscore_support(y_t, y_p, labels=['PS','S','VS','I'])
    acc = accuracy_score(y_t, y_p)
    bacc = balanced_accuracy_score(y_t, y_p)
    
    print('prec: ', p )
    print('rec: ', r)
    print('f1: ', f1)
    print('acc: ', acc)
    print('bacc: ', bacc)

def binary_classify(threshold):
    doc_labels, data = load_data()
    infile = open('newfuzzysim.pickle','rb')
    df = pickle.load(infile)
    values = df.values
    classes = []
    binc = []
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            doc_i = doc_labels[i][:33]
            doc_j = doc_labels[j][:33]
            val = values[i,j]
            #print(val)
            c = 0 
            if val >= 80:
                if val > threshold:
                    c = 1
                else:
                    c = 0

                classes.append([doc_i, doc_j, c])
                binc.append(c)

    result = pd.DataFrame(np.array(classes))
    output = np.array(binc)
    #result.to_excel("classified_TCpairs.xlsx")
    #visualize_dict(prod_dict)
    return output

def binary_test():

    classes = classify()
    df = pd.read_excel(r'excelfiles/evaluations.xlsx')
    cval = classes.values
    dval = df.values
    
    y_true = dval[:,2]
    y_pred = cval[:100,2]
    y_t = np.delete(y_true, [75, 92])
    y_p = np.delete(y_pred, [75, 92])

    y_bt = np.zeros(y_t.shape)
    y_bp = np.zeros(y_p.shape)

    for i,c in enumerate(y_t):
        if c == 'I' or c == 'VS':
            y_bt[i] = 1
        else:
            y_bt[i] = -1

    for i,c in enumerate(y_p):
        if c == 'I' or c == 'VS':
            y_bp[i] = 1
        else:
            y_bp[i] = -1

    print(type(y_bt), y_bp)
    p, r, f1, supp = precision_recall_fscore_support(y_bt, y_bp)
    acc = accuracy_score(y_bt, y_bp)
    bacc = balanced_accuracy_score(y_bt, y_bp)
    #auc = roc_auc_score(y_t, y_p)
    print('prec: ', p )
    print('rec: ', r)
    print('f1: ', f1)
    print('acc: ', acc)
    print('bacc: ', bacc)
    #print('auc: ', auc ) 

def return_binary():
    df = pd.read_excel(r'excelfiles/evaluations.xlsx')
    dval = df.values
    y_true = dval[:,2]
    #y_t = np.delete(y_true, [75, 92])
    y_bt = np.zeros(y_true.shape)

    for i,c in enumerate(y_true):
        if c == 'I' or c == 'VS':
            y_bt[i] = 1
        else:
            y_bt[i] = 0
    return y_bt

def fscore_curve():
    thresholds = np.linspace(80,100,20)
    print(thresholds)
    fscores = []
    y_true = return_binary()

    for t in thresholds:
        pred = binary_classify(t)
        #cval = pred.values
        y_pred = pred[:100]
        #y_pred = np.delete(y_p, [75, 92])
        #print(type(y_pred[0]))
        #print(type(y_true),type(y_pred))
        p, r, f1, supp = precision_recall_fscore_support(y_true, y_pred)
        fscores.append(f1)

    #print(fscores)
    fvec = np.array(fscores)
    avg_fscore = (fvec[:,0] + fvec[:,1]) / 2
    plt.plot(thresholds/100,fvec[:,0], linewidth=3, color='orangered')
    plt.plot(thresholds/100,fvec[:,1], linewidth=3, color='limegreen')
    #plt.plot(thresholds/100,avg_fscore, linewidth=1, linestyle='--')
    plt.legend(ncol=2,labels=['Non-Similar', 'Similar'],bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    #plt.legend(labels=['Non-Similar', 'Similar', 'Average'])
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')




    plt.show()

if __name__ == "__main__":
    '''
    
    fuzzyout, distout = fuzzy_match()
    #print(fuzzyout)
    outfile = open('newfuzzysim.pickle','wb')
    distfile = open('distances.pickle','wb')
    pickle.dump(100*fuzzyout,outfile)
    pickle.dump(distout, distfile)
    outfile.close()
    distfile.close()
    '''
    #visualize('bar')
    #classify()
    #binary_test()
    fscore_curve()
    #print(random_undersample())
    #f = fuzzy_match(print_mode=True)
    #fuzzyout = fuzzy_match()

    #doc_labels, data = load_data()
    #print(data[8])
    #linklist = check_for_links(doc_labels, data)
    #print(linklist)