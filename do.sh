for corpus in sotu ye
do

    for model in word t chi freq
    do

        ./do_import.sh $corpus $model

        for n_topic in 10 50 100
        do

            mkdir corpus/$corpus/$model/$n_topic
            for round in 0 1 2
            do
            
                mkdir corpus/$corpus/$model/$n_topic/$round
                ./do_train.sh $corpus $model $n_topic $round

            done

        done

    done

done