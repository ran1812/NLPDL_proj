# NLPDL_project:领域特定文本

2022年秋NLPDL课程project：领域特定文本-生物领域

## Environment setup

```
transformers
wandb
evaluate
datasets
torch
numpy
```

## Pretrained model and data

预训练模型与预训练数据可以从这里下载：https://drive.google.com/drive/folders/1VBSww78iPc-5lBfV3cSgS9gteMEf1u5b?usp=sharing

其中有三个预训练模型：posttrain, posttrain_clean(+数据清洗), posttrain_adapter(+adapter),将其置于主文件夹下即可

预训练数据为data/posttrain.txt, 将其置于datasets文件夹下即可。

训练数据为生物领域任务bioasq与chemprot，数据来源https://drive.google.com/drive/folders/1Rc_15j3VwnFChzzKj21qIw9lG1UmlBOn?usp=share_link ，将bioasq,chemprot文件夹置于datasets文件夹下即可

## Run

预训练数据的具体下载、解压可以通过get_data.sh完成
```
bash get_data.sh
```

将预训练数据聚合为txt的代码在get_posttrain.ipynb中，预训练数据的清洗代码在clean.ipynb中，依次执行即可。

预训练共三种类型的模型，如下：

运行预训练的代码如下：
```
python run_mlm.py --model_name_or_path roberta-base --output_dir ./posttrain --dataset_name pretrain --per_device_train_batch_size 96 --do_train --overwrite_output_dir
```

其中 --output_dir 可选择 ./posttrain ; ./posttrain_clean ;  ./posttrain_adapter ; 其中dataset_name可选择pretrain (对应第一种output_dir) ; pretrain_clean (对应后两种output_dir)


运行fine-tune的代码如下：
```
python train.py --do_predict --model_name_or_path './posttrain_clean' --output_dir /tmp/ --dataset_name bioasq --do_train --overwrite_output_dir --seed 1 --num_train_epoch 4 
```
其中 --dataset_name 可选择bioasq,chemprot; --output_dir可自行修改为任意输出位置，--seed可自行修改以进行多次测量，我选择的是default(42),1,2,3,4进行测量，--num_train_epoch 可自行修改。


