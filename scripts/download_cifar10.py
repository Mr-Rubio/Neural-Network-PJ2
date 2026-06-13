"""Download and extract CIFAR-10 dataset with mirror fallback."""

import os
import sys
import tarfile
import urllib.request

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HOME, 'data')
TAR_PATH = os.path.join(DATA_DIR, 'cifar-10-python.tar.gz')
EXTRACT_DIR = os.path.join(DATA_DIR, 'cifar-10-batches-py')
EXPECTED_SIZE = 162_000_000  # ~162 MB

MIRRORS = [
    'https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz',
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
]


def download_with_progress(url, dest):
    print(f'Downloading from {url}')
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 / total_size)
            mb = block_num * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f'\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isdir(EXTRACT_DIR) and os.path.isfile(
            os.path.join(EXTRACT_DIR, 'data_batch_1')):
        print(f'CIFAR-10 already ready at {EXTRACT_DIR}')
        return

    if os.path.isfile(TAR_PATH) and os.path.getsize(TAR_PATH) >= EXPECTED_SIZE:
        print(f'Using existing archive: {TAR_PATH}')
    else:
        if os.path.isfile(TAR_PATH):
            os.remove(TAR_PATH)
        for url in MIRRORS:
            try:
                download_with_progress(url, TAR_PATH)
                if os.path.getsize(TAR_PATH) >= EXPECTED_SIZE:
                    print(f'Download complete: {os.path.getsize(TAR_PATH)} bytes')
                    break
                print('File too small, trying next mirror...')
                os.remove(TAR_PATH)
            except Exception as e:
                print(f'Failed: {e}')
                if os.path.isfile(TAR_PATH):
                    os.remove(TAR_PATH)
        else:
            raise RuntimeError('All mirrors failed. Try manual download.')

    print('Extracting ...')
    with tarfile.open(TAR_PATH, 'r:gz') as tar:
        tar.extractall(DATA_DIR)
    print(f'CIFAR-10 ready at {EXTRACT_DIR}')


if __name__ == '__main__':
    main()
