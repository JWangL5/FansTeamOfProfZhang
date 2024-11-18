'''
目前的数据仍旧存在一定的bug
- switchable的荧光蛋白有多个状态，其序列相同，但参数不同，需要考虑这一类蛋白的数据如何做整理
- 有一部分蛋白的序列有genebank ID，但是FPBase中没有直接的sequence记录，需要重新整理
'''

import json
import requests
import pandas as pd
from tqdm import tqdm

# 从FPbase的API中查询荧光蛋白信息，并保存json格式的数据
def get_protein(fpname, download=True, log='./log/fp_download.txt'):
    fpname_ = fpname.replace('/', '-')
    try:
        url = f"https://www.fpbase.org/api/proteins/?name__iexact={fpname}&format=json"
        response = requests.get(url).content
        fp = json.loads(response)[0]
    except:
        with open(log, 'a') as f:
            f.write(f'{fpname_} errored\n')
        return None
    if download:
        with open(f"./data/fp/{fpname_}.json", "w") as f:
            json.dump(fp, f, indent=4)
    return fp

# 从json格式数据中以fasta格式保存出序列信息
def generate_fasta(json, filepath='./data/fpseq.fasta'):
    with open(filepath, 'a+') as f:
        names = [i[1:] for i in f.readlines() if i.startswith('>')]
        if json['name'] not in names and json['seq']!='None':
            f.write(f">{json['name']}\n")
            f.write(f"{json['seq']}\n\n")
            
def clear_fasta(fasta_file_path, log=None, out=None):
    with open(fasta_file_path, 'r') as f:
        flines = [i for i in f.readlines()]
        names = [j[1:] for i,j in enumerate(flines) if j.startswith('>') and flines[i+1]=='None\n']
    
    if log != None:    
        with open(log, 'w') as f:
            for i in set(names):
                f.writelines(i)
            
    if out != None:
        with open(out, 'w') as f:
            # for index in range(len(flines)):
            index = 0
            while index < len(flines):
                if flines[index].startswith('>') and flines[index][1:] in names:
                    print(index, flines[index])
                    index +=3
                else:
                    f.write(flines[index])
                    index +=1
    return names

def load_fasta(fasta_file_path):
    with open(fasta_file_path, 'r') as f:
        lines = [i.strip() for i in f.readlines()]
        res = {item[1:]:lines[index+1] for index, item in enumerate(lines) if item.startswith('>')}
    return res

if __name__=="__main__":
    # 从大表格中逐一下载蛋白质序列
    # data = pd.read_csv("./data/FPsProperty.csv")
    # for i in tqdm(data.iterrows()):
    #     fp = get_protein(i[1]['Name'])
    #     if fp is not None:
    #         generate_fasta(fp)
    
    # 从第一次的结果中补漏
    # with open("./log/fp_download.txt", "r") as f:
    #     for i in f.readlines():
    #         fp = get_protein(i.split(' ')[0], out='./log/fp_without_seq.txt', log='./log/fp_download_2.txt')
    #         if fp is not None:
    #             generate_fasta(fp)
    
    # 获取序列结果为None的数据
    print(clear_fasta('./data/fpseq.fasta', out='./data/fpseq_filtered.fasta'))
    