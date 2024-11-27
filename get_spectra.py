import os
import json
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from get_fasta import load_fasta
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def get_spectra():
    spectra_path = './data/spectrum.json'
    if os.path.isfile(spectra_path):
        with open(spectra_path, 'r') as f:
            fp = json.load(f)
        return fp
    else:
        url = f"https://www.fpbase.org/api/proteins/spectra/"
        response = requests.get(url).content
        fp = json.loads(response)
        with open(f"./data/spectrum.json", "w") as f:
            json.dump(fp, f, indent=4)
        return fp


def process_spectra_data(data, low=0, high=1800):
    spectra = np.zeros(high-low+1)
    for j in data:
        spectra[int(j[0])-low] = j[1]
    return spectra


# 波长到颜色的映射函数
def wavelength_to_rgb(wavelength):
    if 380 <= wavelength <= 440:
        R, G, B = -(wavelength - 440) / (440 - 380), 0.0, 1.0
    elif 440 < wavelength <= 490:
        R, G, B = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif 490 < wavelength <= 510:
        R, G, B = 0.0, 1.0, -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        R, G, B = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif 580 < wavelength <= 645:
        R, G, B = 1.0, -(wavelength - 645) / (645 - 580), 0.0
    elif 645 < wavelength <= 780:
        R, G, B = 1.0, 0.0, 0.0
    else:
        R = G = B = 0.0

    # Adjust intensity for wavelengths outside of visible range
    if wavelength < 380 or wavelength > 780:
        S = 0.0
    elif 380 <= wavelength <= 420:
        S = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 645 < wavelength <= 780:
        S = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
    else:
        S = 1.0

    R, G, B = round(R * S, 3), round(G * S, 3), round(B * S, 3)
    return (R, G, B)


def plot_spectrum(spectrum):
    range = np.array(spectrum['default_ex']) + np.array(spectrum['default_em'])
    start = next((i for i, x in enumerate(range) if x != 0), None) - 50
    end = next((i for i, x in enumerate(reversed(range)) if x != 0), None)
    if start is not None and end is not None:
        end = len(range) - 1 - end + 50
    wavelengths = np.arange(start, end)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelengths, spectrum['default_ex'][start:end], label="Excitation", color="red", linewidth=2)
    ax.plot(wavelengths, spectrum['default_em'][start:end], label="Emission", color="blue", linewidth=2)
    for wl in wavelengths:
        color = wavelength_to_rgb(wl)
        rect = Rectangle((wl, 0), 1, 0.03, color=color, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)
    ax.set_title(f"Excitation and Emission Spectrum for {spectrum['name']}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative Intensity")
    ax.legend(frameon=False)
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1.05)
    plt.show()  


if __name__=="__main__":
    spectrum = get_spectra()
    res = []
    for i in tqdm(spectrum):
        spectra = {'name': i['name']}
        spectra.update({j['state'].lower():process_spectra_data(j['data']) for j in i['spectra']})
        res.append(spectra)
    data = pd.DataFrame(res)
    data = data[['name', 'default_ex', 'default_em']].dropna()
    data.to_json('./data/spectrum_extracted.json', orient="records", lines=True)

    # data = pd.read_json('../data/spectrum_extracted.json', orient="records", lines=True)
    for i in data.iterrows():
        plot_spectrum(i[1])
        break