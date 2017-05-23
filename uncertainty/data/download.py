from urllib.request import urlretrieve
from urllib.parse import urljoin

DATASETS = ['abstracts', 'bmc', 'fly', 'hbc', 'news', 'wiki', 'merged_data']
URL = 'http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/'


def download():
    print('Download Training Data Files')
    for (index, dataset) in enumerate(DATASETS):
        url = urljoin(URL, dataset)
        urlretrieve(url, dataset)
        print('  [{}/{}] {}'.format(index + 1, len(DATASETS), dataset))

if __name__ == "__main__":
    download()
