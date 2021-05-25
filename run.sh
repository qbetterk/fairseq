#!/bin/bash
#
set -xue

export CUDA_VISIBLE_DEVICES=1

text=data/MULTIWOZ2.2
output_dir=data/MULTIWOZ2.2

src=text
tgt=slot

model_dir=checkpoint/checkpoint_multiwoz
# model_dir=checkpoint/checkpoint_multiwoz_all_llf3
# data_dir=${output_dir}/data-bin-all
data_dir=${output_dir}/data-bin

text=data/MULTIWOZ2.2/sep_com # separate comma: value ,
data_dir=${output_dir}/data-bin-sep
model_dir=checkpoint/checkpoint_multiwoz_sep

# fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
#                      --trainpref $text/train --validpref $text/valid \
#                      --testpref $text/test --destdir ${data_dir} \
#                      --workers 60 --joined-dictionary \
#                      --srcdict ${data_dir}/dict.text.txt

data_dir=${output_dir}/data-bin-sep-dict # separate comma and saparate dictionary
model_dir=checkpoint/checkpoint_multiwoz_sep_dict

data_dir=${output_dir}/data-bin-sep-dict-sar # add <\s> during preprocess
model_dir=checkpoint/checkpoint_multiwoz_sep_dict_sar
# fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
#                      --trainpref $text/train --validpref $text/valid \
#                      --testpref $text/test --destdir ${data_dir} \
#                      --workers 100 \
#                      --srcdict ${output_dir}/data-bin-sep-dict/dict.text.txt \
#                      --tgtdict ${output_dir}/data-bin-sep-dict/dict.slot.txt \
#                      --sar


# mkdir -p ${model_dir}
# fairseq-train \
#     ${data_dir} \
#     --ddp-backend=legacy_ddp \
#     --task sar_dst \
#     --criterion nat_loss \
#     --arch cmlm_transformer \
#     --noise random_mask \
#     --optimizer adam --adam-betas '(0.9,0.98)' \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --stop-min-lr '1e-09' --warmup-updates 10 \
#     --warmup-init-lr '1e-07' --label-smoothing 0.1 \
#     --dropout 0.3 --weight-decay 0.01 \
#     --decoder-learned-pos \
#     --encoder-learned-pos \
#     --apply-bert-init \
#     --fixed-validation-seed 7 \
#     --max-tokens 8000 \
#     --reset-optimizer \
#     --fp16 \
#     --max-update 300000 \
#     --seed 0 \
#     --save-dir ${model_dir} \
#     --batch-size 512 \
#     --best-checkpoint-metric length \
#     --max-epoch 80 --save-interval 5 \
#     --length-loss-factor 1.0 \
#     --tensorboard-logdir ${model_dir} \
#     --encoder-layers 6 \
#     --decoder-layers 6 \
#     --encoder-embed-dim 768 \
#     --decoder-embed-dim 768 \
#     --encoder-ffn-embed-dim 3072 \
#     --decoder-ffn-embed-dim 3072 \
#     --encoder-attention-heads 16 \
#     --decoder-attention-heads 16 # 2>&1 | tee ${model_dir}/train.log
#     # --save-interval 5 #2>&1 | tee ${model_dir}/train.log

# fairseq-generate \
#     ${data_dir} \
#     --gen-subset test \
#     --task translation_lev \
#     --path ${model_dir}/checkpoint_best.pt \
#     --iter-decode-max-iter 9 \
#     --iter-decode-eos-penalty 0 \
#     --beam 3 --remove-bpe --print-step \
#     --batch-size 256 --quiet --results-path ${model_dir}/output


text=data/MULTIWOZ2.2
# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # BART large
# # # # # # # # # # # # # # # # # # # # # # # #

output_dir=data/MULTIWOZ2.2_bart
model_dir=checkpoint/checkpoint_multiwoz_bart

# # # # # # # # # # # # # # # BPE preprocess
# for SPLIT in train valid test
# do
#   for LANG in text slot
#   do
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json data/encoder.json \
#     --vocab-bpe data/vocab.bpe \
#     --inputs "$text/$SPLIT.$LANG" \
#     --outputs "$output_dir/$SPLIT.bpe.$LANG" \
#     --workers 1 \
#     --keep-empty;
#   done
# done

# # # # # # # # # # # # # # # binarize dataset
# fairseq-preprocess \
#   --source-lang "text" \
#   --target-lang "slot" \
#   --trainpref "${output_dir}/train.bpe" \
#   --validpref "${output_dir}/valid.bpe" \
#   --testpref "${output_dir}/test.bpe" \
#   --destdir "${output_dir}/data-bin/" \
#   --workers 60 \
#   --srcdict data/dict.txt \
#   --tgtdict data/dict.txt

# fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
#                      --trainpref $text/train --validpref $text/valid \
#                      --testpref $text/test --destdir ${output_dir}/data-bin2 \
#                      --workers 60 --srcdict data/dict_token.txt --tgtdict data/dict_token.txt

# fairseq-train \
#     ${output_dir}/data-bin \
#     --ddp-backend=c10d \
#     --task translation_lev \
#     --criterion nat_loss \
#     --arch cmlm_transformer \
#     --noise random_mask \
#     --share-all-embeddings \
#     --layernorm-embedding \
#     --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-08 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --stop-min-lr '1e-09' --warmup-updates 10 \
#     --warmup-init-lr '1e-07' --label-smoothing 0.1 \
#     --dropout 0.3 --weight-decay 0.01 \
#     --decoder-learned-pos \
#     --encoder-learned-pos \
#     --apply-bert-init \
#     --fixed-validation-seed 7 \
#     --max-tokens 8000 \
#     --max-update 300000 \
#     --seed 0 \
#     --fp16 \
#     --save-dir ${model_dir} \
#     --batch-size 256 \
#     --max-epoch 20 --save-interval 2 \
#     --encoder-layers 12 \
#     --decoder-layers 12 \
#     --encoder-embed-dim 1024 \
#     --decoder-embed-dim 1024 \
#     --encoder-ffn-embed-dim 4096 \
#     --decoder-ffn-embed-dim 4096 \
#     --encoder-attention-heads 16 \
#     --decoder-attention-heads 16 \
#     --restore-file ../pretrain/bart.large/model.pt \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --length-loss-factor 0.1 2>&1 | tee ${model_dir}/train.log

# model_dir=checkpoint_multiwoz_bart_IR
# fairseq-train \
#     ${output_dir}/data-bin \
#     --ddp-backend=legacy_ddp \
#     --task translation_lev \
#     --criterion nat_loss \
#     --arch iterative_nonautoregressive_transformer \
#     --noise full_mask \
#     --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9,0.98)' \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --stop-min-lr '1e-09' --warmup-updates 10000 \
#     --warmup-init-lr '1e-07' --label-smoothing 0.1 \
#     --dropout 0.3 --weight-decay 0.01 \
#     --decoder-learned-pos \
#     --encoder-learned-pos \
#     --pred-length-offset \
#     --length-loss-factor 0.1 \
#     --train-step 4 \
#     --dae-ratio 0.5 \
#     --stochastic-approx \
#     --log-format 'simple' --log-interval 100 \
#     --fixed-validation-seed 7 \
#     --max-tokens 8000 \
#     --max-update 300000 \
#     --seed 0 \
#     --fp16 \
#     --save-dir ${model_dir} \
#     --batch-size 8 \
#     --max-epoch 10 --save-interval 5 \
#     --encoder-layers 12 \
#     --decoder-layers 12 \
#     --encoder-embed-dim 1024 \
#     --decoder-embed-dim 1024 \
#     --encoder-ffn-embed-dim 4096 \
#     --decoder-ffn-embed-dim 4096 \
#     --encoder-attention-heads 16 \
#     --decoder-attention-heads 16 \
#     --restore-file ../pretrain/bart.large/model.pt \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --log-format json 2>&1 | tee ${model_dir}/train.log




# fairseq-generate \
#     ${output_dir}/data-bin \
#     --gen-subset test \
#     --task translation_lev \
#     --path ${model_dir}/checkpoint_best.pt \
#     --iter-decode-max-iter 9 \
#     --iter-decode-eos-penalty 0 \
#     --iter-decode-with-beam 3 \
#     --beam 3 --remove-bpe \
#     --batch-size 64 --quiet --results-path ${model_dir}/output




# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # BART base
# # # # # # # # # # # # # # # # # # # # # # # #

output_dir=data/MULTIWOZ2.2_bart
data_dir=${output_dir}/data-bin-base-sar

# # # # # # # # # # # # # # # # BPE preprocess
# # # # May12 add one space before each slot line
# # # # May13 p.m. --> pm
# # # #       ., --> ,
# for SPLIT in train valid test
# do
#   for LANG in text slot
#   do
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json data/encoder.json \
#     --vocab-bpe data/vocab.bpe \
#     --inputs "$text/$SPLIT.$LANG" \
#     --outputs "$output_dir/$SPLIT.bpe.$LANG" \
#     --workers 200 \
#     --keep-empty;
#   done
# done

# # # # # # # # # # # # # # # binarize dataset
# fairseq-preprocess \
#   --source-lang "text" \
#   --target-lang "slot" \
#   --trainpref ${output_dir}/train.bpe \
#   --validpref ${output_dir}/valid.bpe \
#   --testpref ${output_dir}/test.bpe \
#   --destdir ${data_dir} \
#   --workers 1 \
#   --srcdict ../pretrain/bart.base/dict.txt \
#   --tgtdict ../pretrain/bart.base/dict.txt \
#   --sar

export CUDA_VISIBLE_DEVICES=1
model_dir=checkpoint/checkpoint_multiwoz_bart_sar_pos_keep_enc
model_dir=checkpoint/checkpoint_multiwoz_bart_sar_pos
# model_dir=checkpoint/checkpoint_multiwoz_bart_sar
# model_dir=checkpoint/checkpoint_multiwoz

mkdir -p $model_dir
fairseq-train \
    ${data_dir} \
    --ddp-backend=legacy_ddp \
    --task sar_dst \
    --criterion nat_loss \
    --arch cmlm_transformer \
    --noise random_mask \
    --share-all-embeddings \
    --layernorm-embedding \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-08 \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --max-update 300000 \
    --seed 0 \
    --fp16 \
    --log-interval 10 \
    --save-dir ${model_dir} \
    --batch-size 4 \
    --max-epoch 81 --save-interval 5 \
    --tensorboard-logdir ${model_dir} \
    --encoder-layers 6 \
    --decoder-layers 6 \
    --encoder-embed-dim 768 \
    --decoder-embed-dim 768 \
    --encoder-ffn-embed-dim 3072 \
    --decoder-ffn-embed-dim 3072 \
    --encoder-attention-heads 16 \
    --decoder-attention-heads 16 \
    --length-loss-factor 1.0 \
    --restore-file ${model_dir}/checkpoint_best.pt
    # --encoder-layers-to-keep 6 \
    # --restore-file ../pretrain/bart.base/model.pt \
    # --reset-optimizer --reset-dataloader --reset-meters \
    # --length-loss-factor 0.1 2>&1 | tee ${model_dir}/train.log

# # # # export CUDA_LAUNCH_BLOCKING=1
# fairseq-generate \
#     ${data_dir} \
#     --gen-subset test \
#     --task sar_dst \
#     --path ${model_dir}/checkpoint_last.pt \
#     --iter-decode-max-iter 0 \
#     --iter-decode-eos-penalty 0 \
#     --iter-decode-with-beam 1 \
#     --beam 5 --remove-bpe \
#     --batch-size 4 --results-path ${model_dir}/output --bpe gpt2 --quiet

# # # # # # # # # # Using AR 
# model_dir=checkpoint/checkpoint_multiwoz_bart_base_ar2
# mkdir -p $model_dir
# fairseq-train \
#     ${data_dir} \
#     --ddp-backend=c10d \
#     --task sar_dst \
#     --criterion label_smoothed_cross_entropy \
#     --arch bart_base \
#     --share-all-embeddings \
#     --layernorm-embedding \
#     --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-08 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --stop-min-lr '1e-09' --warmup-updates 10 \
#     --warmup-init-lr '1e-07' --label-smoothing 0.1 \
#     --dropout 0.1 --weight-decay 0.01 \
#     --decoder-learned-pos \
#     --encoder-learned-pos \
#     --fixed-validation-seed 7 \
#     --max-tokens 8000 \
#     --max-update 300000 \
#     --seed 0 \
#     --fp16 \
#     --save-dir ${model_dir} \
#     --batch-size 4 \
#     --max-epoch 1 --no-epoch-checkpoints \
#     --tensorboard-logdir ${model_dir} \
#     --encoder-layers 6 \
#     --decoder-layers 6 \
#     --encoder-embed-dim 768 \
#     --decoder-embed-dim 768 \
#     --encoder-ffn-embed-dim 3072 \
#     --decoder-ffn-embed-dim 3072 \
#     --encoder-attention-heads 12 \
#     --decoder-attention-heads 12 \
#     --restore-file ../pretrain/bart.base/model.pt \
#     --reset-optimizer --reset-dataloader --reset-meters #2>&1 | tee ${model_dir}/train.log

# fairseq-generate \
#     ${data_dir} \
#     --gen-subset test \
#     --task translation \
#     --path ${model_dir}/checkpoint_best.pt \
#     --iter-decode-max-iter 9 \
#     --iter-decode-eos-penalty 0 \
#     --iter-decode-with-beam 1 \
#     --beam 3 --remove-bpe \
#     --batch-size 64 --quiet --results-path ${model_dir}/output --bpe gpt2

# fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
#     --path data-bin/wmt14.en-fr.fconv-py/model.pt \
#     --beam 1 --batch-size 3 --remove-bpe


