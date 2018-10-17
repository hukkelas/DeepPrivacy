import requests
import os
import random
word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
dir_path = os.path.dirname(os.path.realpath(__file__))
word_path = os.path.join(dir_path, "words.txt")
def download_and_save():
    print("Downloading words from:", word_site)
    response = requests.get(word_site)
    WORDS = response.content.splitlines()
    WORDS = [x.decode("utf-8") + "\n" for x in WORDS]
    with open(word_path, 'w') as f:
        f.writelines(WORDS)
    

def random_word(n=1):
    if not os.path.isfile(word_path):
        download_and_save()
    with open(word_path, 'r') as f:
        words = f.readlines()
    return random.choice(words).strip().lower()

if __name__ == "__main__":
    import time
    t = time.time()
    print("Time used:", time.time() - t)