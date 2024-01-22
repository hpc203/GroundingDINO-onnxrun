在运行程序时，要注意输入的提示词的格式，类别之间以" . "隔开，并且确保类别名称在词典文件
vocab.txt里是存在的，否则可能会检测不到目标的。

如果要导出onnx文件，把export_onnx.py放在https://github.com/wenyi5608/GroundingDINO
里运行就可以生成onnx文件的。这个仓库里的代码跟官方仓库https://github.com/IDEA-Research/GroundingDINO
里的代码的不同之处在于
groundingdino\models\GroundingDINO\groundingdino.py里的forward函数的输入参数不同。

已经导出的onnx文件在百度云盘，下载链接：https://pan.baidu.com/s/1_dDxaSMG2vbw47FJ7FdUUg 
提取码：u6lr
