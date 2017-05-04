from urllib.request import urlretrieve

def download():
    print("Downloading preparsed datasets...")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/abstracts", "abstracts")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/allarticles", "allarticles")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/allbio", "allbio")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/bmc", "bmc")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/fly", "fly")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/hbc", "hbc")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/merged_data", "merged_data")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/news", "news")
    urlretrieve("http://people.rc.rit.edu/~bsm9339/corpora/szeged_uncertainty/wiki", "wiki")
    print("Finished!")

if __name__ == "__main__":
    download()
