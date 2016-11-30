# Tagging-with-HMM.
An implementation of Expectation Maximization for HMM tagging’s.

<b>Description</b>

The algorithm predicts the best tag-sequence given an input sequence. It calculates the initial probability distribution from the train file. Over successive iterations, it learns from the un-supervised data in raw-file. After each iteration, the performance is measured over test data and reported.  No smoothing is performed for the EM algorithm.

There are two ways we can compute the probabilities :

<i>1.Add-lambda probabilities:</i> Just use prob_wordtag(), prob_tagtag() with the desired lambda value. It tries to smooth the distribution by reducing the probability of seen sequences and giving non-zero probability to unseen/novel sequences.
Set lambda to 0, to get naïve unsmoothed probability directly estimated from counts. Beware that in such a case, the code would not work where test file / raw file has unseen words.

<i>2.Add-lambda probability with backoff-smoothing :</i> We implement one count smoothing. To use this, instead of prob_wordtag() and prob_tagtag() – use add1_prob_wordtag() and add1_prob_tagtag() with appropriate lambda.

You can change the number of iterations inside the code by the variable num_epochs

<b>Data Format</b>
The program expects a simple \<word\>/\<tag\> as its input. The raw file only contains the \<word\>’s. 
Input sequences are demarcated by ‘###/###’ . 

<b>Program Inputs</b>
The program takes three files as inputs
1. Train File : Used to calculate the initial probabilities. Contains word and its tag pairs.

2.  TestFile : Contains the test data. Used to report performance after each iteration. Contains word and its correct tag pairs.

3. RawFile : Contains the unlabeled raw data. After every iteration, EM algorithm tries to maximize the distribution to this raw file.

<b>Ouputs</b>
We use Viterbi decoding to find the best tag sequence for the test data. After every iteration, we report the tagging accuracy on test data as well as Perplexity per word over the test data and raw data.

<b>Requirements</b>
Python (>=2.7)

<b>To Run</b>

<code>python HMMgem.py \<train_file\> \<test_file\> \<raw_file\> </code>

<b>For More Details:</b>
The code is an attempt to algorithmically simulate  and generalize  what was mentioned in this following paper.
Refer “Eisner, J. (2002). An interactive spreadsheet for teaching the forward-backward algorithm. In Proceedings of the ACL Workshop on Effective Tools and Methodologies for Teaching NLP and CL, pp. 10–18” 

Link to his related lecture : https://vimeo.com/31374528
