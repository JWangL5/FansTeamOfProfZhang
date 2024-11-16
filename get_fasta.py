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
def get_protein(fpname, download=True):
    fpname_ = fpname.replace('/', '-')
    try:
        url = f"https://www.fpbase.org/api/proteins/?name__iexact={fpname}&format=json"
        response = requests.get(url).content
        fp = json.loads(response)[0]
    except:
        with open('./log/fp_download.txt', 'a') as f:
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
        if json['name'] not in names:
            f.write(f">{json['name']}\n")
            f.write(f"{json['seq']}\n\n")
            

if __name__=="__main__":
    data = pd.read_csv("./data/FPsProperty.csv")
    for i in tqdm(data.iterrows()):
        fp = get_protein(i[1]['Name'])
        if fp is not None:
            generate_fasta(fp)
