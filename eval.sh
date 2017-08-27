perl bin/eval.pl -q -b -g data/PTBYM/test_PTBYM.conll.correct \
                       -s saves/mymodel_PTBYM/test_PTBYM.conll.correct \
                       -o saves/mymodel_PTBYM/test_PTBYM.conll.scores.txt



perl bin/eval.pl -q -b -g data/PTB/test_sd.conll \
                       -s saves/mymodel/test_sd.conll \
                       -o saves/mymodel/test_sd.scores.txt

perl bin/eval.pl -q -b -g data/PTB/dev_sd.conll \
                       -s saves/mymodel/dev_sd.conll \
                       -o saves/mymodel/dev_sd.scores.txt
