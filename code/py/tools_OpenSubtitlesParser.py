import xml.etree.ElementTree as ET
import operator
import os,re
from datetime import timedelta
import linecache

def time2sec(string):
    """ This function is used for converting string of time stamp in subtitle into seconds."""
    string = string.replace(' ','')
    d = list(map(float,re.split('\.|,|;|:',string)))
    if len(d) > 3: # the format of time stamp is not unified, some of them have milli-second
        t = timedelta(hours = d[0],minutes = d[1],seconds = d[2]+d[3]/1000)
    else:
        t = timedelta(hours = d[0],minutes = d[1],seconds = d[2])
    return(t.total_seconds())


def multiple_replace(dict_, text):
    """Replace text with multi-words in a dictionary."""
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict_.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict_[mo.string[mo.start():mo.end()]], text) 

dict_sub = {'\"':'\'','´':'\'',
            'n \'t':' n\'t', 'in \'':'ing', 'ing \'':'ing'}


def OpenSubtitleSentence(file_path,dict_=dict_sub):
    # import data
    tree = ET.parse(file_path)
    root = tree.getroot()
    time_list = []
    s_list = []
    s_tmp = ''
    flag_s = 0
    flag_e = 0
    flag_m = 0
    
    # get duration of input document
    for sub in root.iter('subtitle'):
        for duration in sub.iter('duration'):
            duration_time = time2sec(duration.text.split(',')[0])    
    # if the duration is too short or too long we ignore this document
    if (duration_time < 3600*0.5) | (duration_time > 3600*3):
        return [],[]
        
    for child in root:
        if child.tag != 's':
            continue
    
        # get the list of start and end time of each sentence
        # if a movie is in 'E-S' pattern in the whole dialog then time_list = []
        time_tmp = []
        for time in child.iter(tag='time'):
            # time_tmp.append(time.attrib['value'])
            time_tmp.extend([time.attrib['id'],time.attrib['value']])
        
        if not time_tmp:
            flag_m = 1
        elif (time_tmp[0].endswith('S')) & (time_tmp[-2].endswith('E')):
            time_list.extend([time_tmp[1],time_tmp[-1]])
        elif (time_tmp[0].endswith('E')) & (time_tmp[-2].endswith('S')):
            flag_m = 1
        elif (time_tmp[0].endswith('S')) & (time_tmp[-2].endswith('S')):
            time_list.extend([time_tmp[1]])
            flag_s = 1
        elif (time_tmp[0].endswith('E')) & (time_tmp[-2].endswith('E')):
            time_list.extend([time_tmp[-1]])
            flag_e = 1
        
        s = [] # sentence in this loop
        if child.itertext(): # sentence in each s tag
            s.append(''.join(child.itertext()))
        s = ' '.join(s[0].split())
        a = s

        if flag_s:
            s_tmp = a
            flag_s = 0
            continue
        
        if flag_m:
            try:
                s_tmp = s_tmp +' ' + a
            except:
                print(child.attrib)
            flag_m = 0
            continue
            
        if flag_e:
            a = s_tmp + ' ' + a
            flag_e = 0
            s_tmp = ''
        
        a = multiple_replace(dict_, a)
        a = multiple_replace(dict_, a)
        a = a.replace('\\','')
        a = re.sub('ca n\'t','can n\'t',a)
        a = re.sub('Ca n\'t','Can n\'t',a)
        
        # remove brackets and contents in the brackets/special symbols
        a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\♪.*?♪|\\#.*?#|\\=.*?=|\\¶.*?¶", "", a)
        a = re.sub('[0-9]+', '<NUM>', a) # replace specific number with NUM
        # a = a.replace('-','')
        a = a.lstrip().rstrip()# remove spaces before or after sentence
            
        if (not len(a)) | (any(re.findall(r' Season.*?Episode |Subtitles|Subtitle | Episode ',a, re.IGNORECASE))) |(not (any(re.findall(r'\.|,|\?|!|\'|\"',a)))): 
            # skip null sentence; delete  '- Season x Episode x -' or 'Subtitles by' lines;
            # delete non-dialog sentences, eg. titles of episode
            del time_list[-2:]
            continue
            
        s_list.append(a)
    # convert to seconds
    time_second = []
    for time in time_list:
        # print(child.attrib,time)
        time_second.append(time2sec(time))
    time = time_second[::2]
    
    return s_list,time

def SaveSentence(save_path,s_,time_):
    n = 1
    new_path = save_path
    while os.path.exists(new_path):
        new_path = save_path.split('.')[0]+'_'+str(n)+'.txt'
        n +=1
    
    f = open(new_path,'a')
    for i in range(len(s_)):
        # to find if short dialogue in one sentence, beginning with '- '
        tmp_sentence = list(filter(None,s_[i].split('- '))) 
        tmp_time = str(time_[i])
        for s in tmp_sentence:
            a = tmp_time + '|' + '<GO>' + s + '<EOS>\n'
            f.write(a)
    return

file = '/Volumes/Files/en/OpenSubtitles/AllFilePath.txt'
f = open(file)
files = [line.rstrip('\n') for line in f]

save_dir = '/Volumes/Files/en/OpenSubtitles/txt/'
for i in range(99000,len(files)):
    try:
        if i % 1000 == 0:
            print(i,i/len(files))
        tmp_path = files[i]    
        s_list,time_ = OpenSubtitleSentence(tmp_path)
        save_path = save_dir+os.path.basename(tmp_path).split('.')[0] + '.txt' 
        if len(s_list) & len(time_): 
    # if s_list and time_ not Null, then save the result
           SaveSentence(save_path,s_list,time_)
    except:
        continue