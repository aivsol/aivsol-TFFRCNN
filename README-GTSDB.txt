# Training
python ./faster_rcnn/train_net.py 
    --gpu 0 
    --weights ./data/pretrain_model/VGG_imagenet.npy 
    --imdb gtsdb_train 
    --iters 70000 
    --cfg  ./experiments/cfgs/faster_rcnn_gtsdb_end2end.yml 
    --network VGGnet_train 
    --restore 0 
    --set EXP_DIR exp_dir

# Demo
python demo_gtsdb.py --model output/default/gtsdb_train/VGGnet_fast_rcnn_iter_10000.ckpt
