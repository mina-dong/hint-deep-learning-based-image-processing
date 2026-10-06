Teacher-only originals (excluded from the student ZIP):
mnist.npz                              [unchanged]
cifar-10-binary.tar.gz                 [or cifar-10-python.tar.gz]
images.tar.gz                         [Oxford-IIIT Pet official images]
annotations.tar.gz                    [Oxford-IIIT Pet official annotations]

Or extracted Oxford images/ and annotations/trainval.txt (also supported under oxford-iiit-pet/).
Do not rename the old cats_and_dogs_filtered.zip to an Oxford archive.
CIFAR prefix cache for R2: cifar_prefix_source_10000.npz (retains 1000/class).
Old R1 prefix cache remains untouched but is insufficient for R2.
Originals can be placed elsewhere by setting SOURCE_DIR in 00_build_data_bundle.py.
Oxford full original download: approximately 800 MB. Only 2000 selected 96x96 images are bundled for students.
Previous data pairs are preserved under bundle_backups/ after successful replacement; this folder is never in the student ZIP.
