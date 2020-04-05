#### News Video Story Segmentation

---

- Data 文件夹：

  包含input, output, GroundTruth文件夹，

  input是语音转写文字后的json文件；output是候选节点提取的特征向量文件；GroundTruth是候选节点的真值

- Tool 文件夹：我自己用来转GBK->UTF8的

- Test.py 主程序调用exfeature.py和meta.py

- meta.py 处理的单个对象，候选节点

- exfeature.py 提取特征