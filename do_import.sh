/home/jin/Mallet/bin/mallet import-dir \
--input corpus/$1/$2/dir_train \
--output corpus/$1/$2/input.mallet \
--keep-sequence \
# --preserve-case \ # German
--token-regex '[\p{L}\p{M}\p{P}]+'

/home/jin/Mallet/bin/mallet import-dir \
--input corpus/$1/$2/dir_test \
--output corpus/$1/$2/test.mallet \
--use-pipe-from corpus/$1/$2/input.mallet
