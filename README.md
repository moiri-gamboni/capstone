# PyForgit
Python 3 Forgit implementation on ENRON email dataset for Capstone at NYUAD

1. download ENRON email dataset (maildir in root)
2. ```pip install marisa-trie```
3. ```emails.preprocess()``` to parse emails and generate trie
4. ```emails.main(sample_size=100, ratio_step=0.1, max_len=50, verbose=False)``` 
  
  generates statistics which are continuously written to *stats.csv*. 
  
  csv header: 
  *'md5',
  'number of tokens',
  'ratio forgotten',
  'email tree edges computed',
  'complete emails compared'*
  
  *sample_size* emails smaller or equal to *max_len* are checked for each *ratio_step* between *ratio_step* and *1-ratio_step*
