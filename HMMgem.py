import argparse
import math
import copy
import pprint
import numpy as np


#Computes the alpha probabilities of the inputs. Alpha probabilities denote the probability of reaching a state from the start. 
def compute_alpha(inputs):
    forward_trellis=[]
    forward_trellis.append([('###',0)])
    len_inputs=len(inputs)
    for x in xrange(1,len_inputs):
        current_word = 'OOV'
        temp_trellis=[]
        if inputs[x] in word_tag:      #a[x] is the current test inputs
            current_word = inputs[x]  
        for ti in word_tag[current_word]:
            alpha = -float("inf")
            for y in xrange(len(forward_trellis[x-1])):
                ti_1 = forward_trellis[x-1][y][0]
                alpha_1 = forward_trellis[x-1][y][1]
                p = prob_tagtag(ti,ti_1,1) + prob_wordtag(inputs[x],ti,1)+alpha_1
                #p = add1_prob_tagtag(ti,ti_1,1+sing_tt[ti_1]) + add1_prob_wordtag(inputs[x],ti,1+sing_tw[ti])+alpha_1
                alpha = logsumexp(alpha,p)
            temp_trellis.append((ti,alpha))
        forward_trellis.append(temp_trellis)
    S = forward_trellis[len(forward_trellis)-1][0][1]
    return forward_trellis,S 

#Computes the beta probabilites of a state. They denote the probability of reaching the end state from the current state.
def compute_beta(inputs):
    # extra ### 
    backward_trellis=[]
    result_reverse = []
    backward_trellis.append([('###',0)])
    len_inputs = len(inputs)
    for x in xrange(len_inputs-2,-1,-1):
        current_word = 'OOV'
        temp_trellis=[]
        if inputs[x] in word_tag:
            current_word = inputs[x]
        for ti in word_tag[current_word]:
            beta = -float("inf")
            for y in xrange(len(backward_trellis[len_inputs - 2 - x])):
                ti_1 = backward_trellis[len_inputs - 2 - x][y][0]
                beta_1 = backward_trellis[len_inputs - 2 - x][y][1]
                p = prob_tagtag(ti_1,ti,1) + prob_wordtag(inputs[x+1],ti_1,1)+beta_1
                #p = add1_prob_tagtag(ti_1,ti,1+sing_tt[ti]) + add1_prob_wordtag(inputs[x+1],ti_1,1+sing_tw[ti_1])+beta_1
                beta = logsumexp(beta,p)
            temp_trellis.append((ti,beta))
        backward_trellis.append(temp_trellis)
    
    k = len(backward_trellis)    
    for i in xrange(k-1,-1,-1):
        result_reverse.append(backward_trellis[i])
    
    return backward_trellis,result_reverse
     


#For every state alpha times beta represents the probability of all paths that pass through that state.
def compute_alpha_beta(alpha_values,beta_values):
    # For each node in trellis, we are storing the joint prob of alpha and beta
    len_beta_values = np.size(beta_values)
    alpha_beta_values=[]
    for i in xrange(len(alpha_values)):
        temp_alpha_beta=[]
        for j in xrange(len(alpha_values[i])):
            if alpha_values[i][j][0] == beta_values[len_beta_values -1 - i][j][0]:
                #adding alpha and beta as values in log 
                temp_alpha_beta.append((alpha_values[i][j][0], alpha_values[i][j][1] + beta_values[len_beta_values -1 - i][j][1]))
        alpha_beta_values.append(temp_alpha_beta)
    return alpha_beta_values

#After computing the alpha-beta values , we use that to  update the probabilites for the next round of EM algorithm.
def get_countsfrom_AlphaBeta(inputs,alpha_beta_values,alpha_values,beta_values,logS):
    dt_new = copy.deepcopy(dt_org)
    d_tt_new = copy.deepcopy(d_tt_org)
    d_tw_new = copy.deepcopy(d_tw_org)
    
    for x in xrange(1,len(alpha_beta_values)):
        curr_word = inputs[x]
        for y in xrange(len(alpha_beta_values[x])):
            curr_tag = alpha_beta_values[x][y][0]
            p = math.exp(alpha_beta_values[x][y][1] - logS)
            
            dt_new[curr_tag] += p
            
            if(str(curr_word)+'/'+str(curr_tag) not in d_tw_new):
                d_tw_new[str(curr_word)+'/'+str(curr_tag)] = p
            else:
                d_tw_new[str(curr_word)+'/'+str(curr_tag)] += p
            
            for z in xrange(len(alpha_beta_values[x-1])):
                tag_prev = alpha_beta_values[x-1][z][0]
                p_tag = math.exp(alpha_values[x-1][z][1] + beta_values[x][y][1] + prob_wordtag(curr_word, curr_tag, 1) + prob_tagtag(curr_tag,tag_prev,1) - logS)
                #p_tag = math.exp(alpha_values[x-1][z][1] + beta_values[x][y][1] + add1_prob_wordtag(curr_word, curr_tag, 1+sing_tw[curr_tag]) + add1_prob_tagtag(curr_tag,tag_prev,1+sing_tt[tag_prev]) - logS)             
                
                if(str(curr_tag)+'/'+str(tag_prev) not in d_tt_new):
                    d_tt_new[str(curr_tag)+'/'+str(tag_prev)] = p_tag
                else:
                    d_tt_new[str(curr_tag)+'/'+str(tag_prev)] += p_tag
    
    return dt_new,d_tt_new,d_tw_new
    
#Runs the forward-backward algorithm
def doFwdBck(a,iter):
    #print '***************FB Starts**************************'    
    pp = pprint.PrettyPrinter(indent=4)
    #print " Computing alpha"
    
    alpha_values,S = compute_alpha(a) # S is the total logprob of going from start to end state. Its a constant
    #pp.pprint(alpha_values)
    #print math.exp(S)
    #pp.pprint(alpha_values)
    #print "*********************************"
    #print " Computing beta"
    beta_values,reverse_beta_values = compute_beta(a)
    
    alpha_beta_values = compute_alpha_beta(alpha_values,beta_values)
    
    
    #pp.pprint(alpha_beta_values)
    give_fwdbck_result(alpha_beta_values,result,iter)
    dt_new,d_tt_new,d_tw_new = get_countsfrom_AlphaBeta(a, alpha_beta_values, alpha_values, reverse_beta_values, S)
    return dt_new, d_tt_new, d_tw_new
    
    
def logsumexp(x, y):
    
    if x == float('-inf'):
        return y
    if y == float('-inf'):
        return x    
    if y <= x:
        return x + math.log(1 + math.exp(y - x))
    else: 
        return y + math.log(1 + math.exp(x - y))

#Only used if we want to use 1-count smoothing for probabilites
def get_singleton_counts():
    
    for tag in dt.keys():
        if tag not in sing_tt:
            sing_tt[tag] = 0
        if tag not in sing_tw:
            sing_tw[tag] = 0  
    
    for word in d_tt.keys():
        wordtag = word.split('/')[1]
        if d_tt[word] == 1:
            sing_tt[wordtag] += 1
                
    for word in d_tw.keys():
        wordtag = word.split('/')[1]
        if d_tw[word] == 1:
            sing_tw[wordtag] += 1
            

def ptt_bckoff(ti, ti_1):
    if ti not in dt:
        return 0
    else:
        return float(dt[ti])/n

def ptw_backoff(wi,ti):
    if wi not in dw:
        return float(1)/(n+V)
    else:
        return float(dw[wi]+1)/(n+V)


#calculates the probability of curr tag given prev. tag. Also called Transition probabilites in HMM 
def prob_tagtag(a,b,lambdas):
    if ( str(str(a)+'/'+str(b)) not in d_tt):
        num = lambdas
    else:
        num = d_tt[str(a)+'/'+str(b)] + lambdas
    
    if num == 0:
        return float('-inf')
    den = dt[b] + distinct_tag_count*lambdas

    return math.log(float(num)/den)


#Calculates the probability of a tag given the input word. Also called Emission probability in HMM.	
def prob_wordtag(a,b,lambdas):
    
    if (a==b and a=='###'):
        return 0
    
    if ( str(str(a)+'/'+str(b)) not in d_tw):
        num = lambdas
    else:
        num = d_tw[str(a)+'/'+str(b)] + lambdas
        
    if num == 0:
        return float('-inf')
    

    else:
        #den = d_tw[b] + unique_word[b]+1
        den = dt[b] + V*lambdas
    return math.log(float(num)/den)

#with 1-count smoothing
def add1_prob_tagtag(a,b,lambdas):
    if ( str(str(a)+'/'+str(b)) not in d_tt):
        c = 0
    else:
        c = d_tt[str(a)+'/'+str(b)]
    
    num = c + lambdas*ptt_bckoff(a, b)
    if num == 0:
        return float('-inf')
    den = dt[b] + lambdas
    return math.log(float(num)/den)

#with 1-count smoothing
def add1_prob_wordtag(a,b,lambdas):
    
    if (a==b and a=='###'):
        return 0
    
    if(str(str(a)+'/'+str(b)) not in d_tw):
        c = 0
    else:
        c = d_tw[str(a)+'/'+str(b)]
    
    
    num = c + lambdas*ptw_backoff(a, b)
    if num == 0:
        return float('-inf')
    
    den = dt[b] + lambdas
    return math.log(float(num)/den)


    
#Runs viterbi decoding to find the best output sequence.
def viterbi_test(a , result):
    b=[]   #stores the HMM
    b.append([('###',-1,0)])
    
    len_a = len(a)
    for x in xrange(1,len_a):      #ignore the first test_word which is ###
        temp = 'OOV'
        bt=[]
        if a[x] in word_tag:      #a[x] is the current test input
            temp = a[x]
            
        for ti in word_tag[temp]:
            mu = -float("inf")
            bk = -1
            for y in xrange(len(b[x-1])):
                ti_1 = b[x-1][y][0]
                mu_1 = b[x-1][y][2]
                temp_mu = prob_tagtag(ti,ti_1,1) + prob_wordtag(a[x],ti,1) + mu_1
                #temp_mu = add1_prob_tagtag(ti,ti_1,1+sing_tt[ti_1]) + add1_prob_wordtag(a[x],ti,1+sing_tw[ti]) + mu_1
                #temp_mu = add1_prob_tagtag(ti,ti_1,0) + add1_prob_wordtag(a[x],ti,0) + mu_1
                if temp_mu > mu:
                    bk = y
                    mu = temp_mu
                    tag = ti

            bt.append((tag,bk,mu))
        b.append(bt)
    

    give_viterbi_results(b,result)

def give_viterbi_results(b,results):
    predict_results = []
    len_b = len(b)-1
    novel_error = 0
    novel_count = 0
    error = 0    
    count = len(b)-2
    seen_error = 0
    seen_count = 0
    #print len(b[len_b])
    y  = 0
    cross_entropy = math.exp(-b[len_b][y][2]/len_b)
    #print 'Cross Entropy is ',fin
    while(len_b >= 0):
        tag = b[len_b][y][0]
        predict_results.append(tag)
        y = b[len_b][y][1]
        len_b  = len_b-1
    
    predict_results.reverse()
    # print predict_results
    
    count = len(results)
    for i  in xrange(0,len(results)):
        if a[i] == '###':        #accuracy doesnt consider the sentence markers
            count -=1
            continue
        
        if a[i] not in unique_train_words:
            if a[i] not in unique_raw_words:
                novel_count += 1
            else:
                seen_count += 1
        
        if(predict_results[i] != results[i]):
            error += 1
            
            if a[i] not in unique_train_words:
                if a[i] not in unique_raw_words:
                    novel_error += 1
                else:
                    seen_error += 1

            
    known_error = error-novel_error-seen_error
    known_count = count-novel_count-seen_count

    if novel_count!=0:
        novel_accuracy = (float)(novel_count-novel_error)*100/novel_count
    else:
        novel_accuracy = 0
    Known_accuracy = (float)(known_count - known_error)*100/known_count
    
    if seen_count != 0:
        Seen_accuracy = float(seen_count-seen_error)*100/seen_count
    else:
        Seen_accuracy = 0
    
    Total_accuracy = float((count-error)*100)/(count)
    
    print 'Tagging accuracy (Viterbi decoding):',Total_accuracy,"% (known:",Known_accuracy,"% seen:",Seen_accuracy,"% novel:",novel_accuracy,'%)'
    print 'Perplexity per Viterbi-tagged test word:',cross_entropy
    
def give_fwdbck_result(alpha_beta_values,results,iter):
    predict_results = []
    S = 0    
    for i in xrange(len(alpha_beta_values)):
        tmp_prob = float('-inf')
        tmp_tag = 'LOL'
        
        for j in xrange(len(alpha_beta_values[i])):
            if alpha_beta_values[i][j][1] > tmp_prob:
                tmp_tag = alpha_beta_values[i][j][0]
                tmp_prob =alpha_beta_values[i][j][1]    
        
        if(len(alpha_beta_values[i]) == 1):
            S = alpha_beta_values[i][0][1]
            #print S
#             for k in xrange(len(alpha_beta_values[i])):
#                 p = math.exp(alpha_beta_values[i][k][1])
#                 S += p
        predict_results.append(tmp_tag)

    print 'Iteration',iter,': Perplexity per untagged raw word:',math.exp(-float(S)/(len(alpha_beta_values)-1))
    #print 'Tagging accuracy (posterior decoding):',Total_accuracy,"% (known:",Known_accuracy,"% novel:",novel_accuracy,'%)'

def debug_counts():
    tag_sum = 0
    for tag in  unique_tags:
        tag_sum += dt[tag]
        #print tag,d_tw[tag]
    
    print 'Sum of tags is',tag_sum
    
    word_tag_sum = 0
    for word in unique_train_words:
        for tag in unique_tags:
            if str(word+'/'+tag) in d_tw:
                word_tag_sum += d_tw[word+'/'+tag]
    
    print 'Sum of Word-tags is',word_tag_sum
    
    tag_tag_sum = 0
    for tag1 in unique_tags:
        for tag2 in unique_tags:
            if str(tag1+'/'+tag2) in d_tt:
                tag_tag_sum += d_tt[tag1+'/'+tag2]
    
    print 'Sum of tag-tags is',tag_tag_sum

def set_to_Zero(dicts):
    for curr_key in dicts.keys():
        dicts[curr_key] = 0

if __name__=="__main__":
    #parse cmd line arguments
    parser = argparse.ArgumentParser(description='General')
    parser.add_argument("train", help="train file")
    parser.add_argument("test", help="test")
    parser.add_argument("raw", help="raw")
    args = parser.parse_args()

    # filter out comments and new lines from grammar
    num_epochs = 11
	f = open(args.train,'r')
    f = [x.replace('\n', '') for x in f]
    # make a dictionary of counts
    len_train = len(f)
    n = len_train-1        #size of training corpus excluding first line
    d_tw={}    #for current  tags,and word_tag sequences
    d_tt = {}  #for current tag-tag sequences
    dw = {}   #for words
    dt = {} #for current tags
    
    d_tw_org={}    #for tags,and word_tag sequences
    d_tt_org = {}  #for tag-tag sequences
    dt_org = {} #for tags
    
    d_tw_new={}    #for tags,and word_tag sequences
    d_tt_new = {}  #for tag-tag sequences
    dt_new = {} #for tags
    
    word_tag  = {}
    unique_tags = set()     #set of unique tags
    unique_train_words = set()    # set of unique words
    unique_raw_words = set()     #set of unique raw words
    sing_tt={}                 # singleton counts of tag-tag-sequences
    sing_tw={}                   #singleton counts of tag-words
    
    
    f[0] = f[0].split('/')
    prev = f[0][1]
    
    for x in xrange(1 , len_train):
        # print x
        element = f[x]
        f[x] = f[x].split('/')                                             
        word_input = f[x][0]
        tag = f[x][1]
        
        if word_input not in unique_train_words:
            unique_train_words.add(word_input)
        
        if tag not in unique_tags:
            unique_tags.add(tag)

        if word_input in word_tag:
            if tag not in word_tag[word_input]:
                word_tag[word_input].append(tag)
        else:
                word_tag[word_input] = [tag]    

        if (element in d_tw):   #add word-tag and tag counts
            d_tw[element] += 1
        else:
            d_tw[element] = 1
        
        if (f[x][1]  in dt):
            dt[f[x][1]] += 1        
        else:
            dt[f[x][1]] = 1
        
        if (word_input in dw):   #add word counts 
            dw[word_input] += 1
        else:
            dw[word_input] = 1
        
        s = str(f[x][1])+'/'+str(prev)
        
        if s in d_tt:
            d_tt[s] +=1
        else:
            d_tt[s] = 1
   

        prev = f[x][1]

    distinct_word_count = len(unique_train_words)
    distinct_tag_count = len(unique_tags)
    word_tag['OOV'] = copy.deepcopy(unique_tags) 
    word_tag['OOV'].remove('###')
    
    #Copy initial training counts into original
    dt_org = copy.deepcopy(dt)
    d_tt_org = copy.deepcopy(d_tt)
    d_tw_org = copy.deepcopy(d_tw)
    
	#Uncomment them if you dont want your initial probabilites to be learnt from train data. They would all be set to 0.
    #set_to_Zero(dt_org)
    #set_to_Zero(d_tt_org)
    #set_to_Zero(d_tw_org)
    
    get_singleton_counts()
    #print "Training Done. Testing ..."

    # testfile
    f = open(args.test,'r')
    f = [x.replace('\n', '') for x in f]

    # make 2 lists for 
    len_test = len(f)
    a=[]   #stores test words
    result=[]  #stores test_result

    
    for x in xrange(len_test):  
        element = f[x]
        f[x] = f[x].split('/')                                             
        a.append(f[x][0])
        result.append(f[x][1])
    
    #rawFile
    f = open(args.raw,'r')
    f = [x.replace('\n', '') for x in f]
    len_raw = len(f)
    raw = []   #stores the raw data
    
    for x in xrange(len_raw):
        element = f[x]
        if element not in dw:
            dw[element] = 1
            unique_raw_words.add(element)
        else:
            dw[element] += 1
        raw.append(element)
    
    #V  = len(unique_train_words) + 1   #this was what it was before
    V = len(dw.keys())+1     #now vocabulary includes total distinct words in raw and train
    
    for i in xrange(num_epochs):
        #print 'n is',n
        if i == 1:
            n += len_raw
        viterbi_test(a, result)    
        dt,d_tt,d_tw = doFwdBck(raw,i)

    
   
