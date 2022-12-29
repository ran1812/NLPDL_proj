# NLPDL_project:领域特定文本

预训练模型与预训练数据可以从这里下载：https://drive.google.com/drive/folders/1VBSww78iPc-5lBfV3cSgS9gteMEf1u5b?usp=sharing

其中有三个预训练模型：posttrain, posttrain_clean(+数据清洗), posttrain_adapter(+adapter),将其置于主文件夹下即可

预训练数据为data/posttrain.txt, 将其置于datasets文件夹下即可。预训练数据的具体获得过程可以通过get_data.sh完成，预训练数据的清洗代码在clean.ipynb中，依次执行即可。

训练数据为生物领域任务bioasq与chemprot，数据来源https://drive.google.com/drive/folders/1Rc_15j3VwnFChzzKj21qIw9lG1UmlBOn?usp=share_link ，将bioasq,chemprot文件夹置于datasets文件夹下即可
