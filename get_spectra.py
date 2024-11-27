import json
import requests
import pandas as pd
from tqdm import tqdm
from get_fasta import load_fasta


def get_spectra(fpname, download=True, log='./log/spectra_download.txt'):
    fpname_ = fpname.replace('/', '-')
    try:
        url = f"https://www.fpbase.org/api/proteins/spectra/?name__iexact={fpname}"
        response = requests.get(url).content
        fp = json.loads(response)[0]
    except:
        with open(log, 'a') as f:
            f.write(f'{fpname} errored\n')
        return None
    if download:
        with open(f"./data/spectrum/{fpname_}.json", "w") as f:
            json.dumps(fp, f, indent=2)
    return fp


def generate_matrix_from_spectra(spfile):
    with open(spfile, 'r') as f:
        sp = json.load(f)
    ex = sp[0]['spectra'][0]
        

if __name__=="__main__":
    fasta = load_fasta('./data/fpseq.fasta')
    name = list(fasta.keys())
    for i in tqdm(name[45:]):
        get_spectra(i)