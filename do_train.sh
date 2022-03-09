/home/jin/Mallet/bin/mallet train-topics  \
--input corpus/$1/$2/input.mallet \
--num-topics $3 \
--output-topic-keys corpus/$1/$2/$3/$4/keys.txt \
--diagnostics-file corpus/$1/$2/$3/$4/diagnostics.xml \
--inferencer-filename corpus/$1/$2/$3/$4/inferencer \
--evaluator-filename corpus/$1/$2/$3/$4/evaluator \
--num-top-words 100 \
--show-topics-interval 1000
# --output-state corpus/$1/$2/$3/$4/topic-state.gz \
# --output-doc-topics corpus/$1/$2/$3/$4/compostion.txt \

/home/jin/Mallet/bin/mallet evaluate-topics \
--evaluator corpus/$1/$2/$3/$4/evaluator \
--input corpus/$1/$2/test.mallet \
--output-prob corpus/$1/$2/$3/$4/prob
# --output-doc-probs corpus/$1/$2/$3/$4/doc-prob \
 

# /home/jin/Mallet/bin/mallet infer-topics \
# --inferencer corpus/$1/$2/$3/$4/inferencer \
# --input corpus/$1/$2/test.mallet \
# --output-doc-topics corpus/$1/$2/$3/$4/doc-topics-test
