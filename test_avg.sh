modelfile=$1  
dataset=$2
num_checkpoints=$3
python3 scripts/average_checkpoints.py --inputs checkpoints/$modelfile --num-epoch-checkpoints $num_checkpoints --output checkpoints/$modelfile/average_epoch.pt 

fairseq-generate data-bin/$dataset --gen-subset test --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test_avg_last_${num_checkpoints}epoch
fairseq-generate data-bin/$dataset --gen-subset test1 --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test1_avg_last_${num_checkpoints}epoch
fairseq-generate data-bin/$dataset --gen-subset test2 --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test2_avg_last_${num_checkpoints}epoch
fairseq-generate data-bin/$dataset --gen-subset test3 --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test3_avg_last_${num_checkpoints}epoch
fairseq-generate data-bin/$dataset --gen-subset test4 --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test4_avg_last_${num_checkpoints}epoch
fairseq-generate data-bin/$dataset --gen-subset test5 --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test5_avg_last_${num_checkpoints}epoch
fairseq-generate data-bin/$dataset --gen-subset test6 --path checkpoints/$modelfile/average_epoch.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json > results/pred_${modelfile}_test6_avg_last_${num_checkpoints}epoch
